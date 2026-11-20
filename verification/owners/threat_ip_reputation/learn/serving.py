# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Gated reference-engine serving and restricted leases on an owned CPU runner."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

import contracts
from class_binding import collect_classes
from remote import prepare
from wire import Server, Backend
import lease_common

HERE = Path(__file__).resolve().parent


def finite_bits(value):
    return type(value) is int and 0 <= value < 1 << 32 and value & 0x7f800000 != 0x7f800000


def ready_report(raw):
    if len(raw) > 1 << 20:
        raise ValueError("serving report byte ceiling")
    report = lease_common.strict_json(raw)
    if (report.get("schema_version") != 1 or report.get("family") != contracts.FAMILY or report.get("status") != "REGISTRATION_READY"
            or report.get("fit_on_audit") is not False or not re.fullmatch(r"[a-f0-9]{40}", report.get("commit", ""))
            or type(report.get("run_id")) is not int or report["run_id"] < 1):
        raise ValueError("serving requires an actual complete independent audit, not native development")
    for field, filename in (("foundation_sha256", "foundation_lora_fp32.onnx"), ("engine_sha256", "foundation_engine.wasm"), ("tokenizer_sha256", "tokenizer.json")):
        if report.get(field) != contracts.FOUNDATION[filename]:
            raise ValueError("serving foundation identity differs")
    for field in ("package_sha256", "package_keccak"):
        if not re.fullmatch(r"[a-f0-9]{64}", report.get(field, "")):
            raise ValueError("literal bare package identity required")
    lock = json.loads((HERE / "candidate-lock.json").read_text())
    if report["package_sha256"] != lock["package_sha256"] or report["package_keccak"] != lock["package_keccak"] or report.get("package_bytes") != lock["package_bytes"]:
        raise ValueError("serving candidate differs from exact locked audited package")
    source, quality, runtime = (report.get(name, {}) for name in ("source", "quality", "runtime"))
    if (source.get("normalized_overlap") != 0 or source.get("audit_used_for_selection") is not False or source.get("grants_verified") is not True
            or source.get("independent_groups", 0) < 32 or not re.fullmatch(r"[a-f0-9]{64}", source.get("source_manifest_sha256", ""))):
        raise ValueError("independent source/grant gate required before serving")
    if quality.get("passed") is not True or quality.get("audit_pairs", 0) < 64 or quality.get("all_supported_formats_tested") is not True:
        raise ValueError("complete all-format quality audit required")
    formats = quality.get("formats", {})
    if set(formats) != contracts.FORMATS:
        raise ValueError("every supported format must pass")
    trained = {fmt: {key: formats[fmt][key] for key in ("accuracy", "per_label")} for fmt in contracts.FORMATS}
    untrained = {fmt: {"accuracy": formats[fmt]["untrained_accuracy"]} for fmt in contracts.FORMATS}
    shortcuts = {fmt: formats[fmt]["shortcut_accuracy"] for fmt in contracts.FORMATS}
    if not contracts.validation_gate(trained, untrained, shortcuts)["passed"]:
        raise ValueError("unchanged quality/label/baseline gate failed")
    if (runtime.get("full_parser_passed") is not True or runtime.get("reference_engine_passed") is not True
            or runtime.get("repeat_bits_identical") is not True or runtime.get("negative_identity_controls", 0) < 3):
        raise ValueError("actual parser/reference/identity proof required")
    vectors = runtime.get("vectors", [])
    if not isinstance(vectors, list) or not 8 <= len(vectors) <= 100:
        raise ValueError("actual reference vector count out of bounds")
    for vector in vectors:
        tokens = vector.get("tokens")
        if (not isinstance(tokens, list) or not 1 <= len(tokens) <= 512 or any(type(token) is not int or not 0 <= token < 50368 for token in tokens)
                or not finite_bits(vector.get("score_bits"))):
            raise ValueError("invalid actual reference vector")
    return report


def lease_row(report, reservation, report_sha):
    if (reservation.get("family") != contracts.FAMILY or reservation.get("package_keccak") != report["package_keccak"]
            or type(reservation.get("slot")) is not int or not 1 <= reservation["slot"] <= 6
            or reservation.get("forward_port") != 48600 + reservation["slot"]
            or reservation.get("ssh_host") != lease_common.HOST or reservation.get("ssh_user") != lease_common.SSH_USER):
        raise ValueError("actual existing free endpoint reservation required")
    expected_url = "https://tga1.zkasuran.dev/_s2o_threat_ip_reputation_" + report["package_keccak"][:8] + "/runnable"
    if reservation.get("runnable_url") != expected_url or not re.fullmatch(r"[a-f0-9]{64}", report_sha):
        raise ValueError("reservation URL/report identity mismatch")
    return {"family": contracts.FAMILY, "owner": lease_common.DEFAULT_OWNER, "slot": reservation["slot"], "forward_port": reservation["forward_port"],
            "runnable_url": expected_url, "keccak": report["package_keccak"], "sha256": report["package_sha256"], "existing_registration_id": 0,
            **{kind + "_proof": {"sha256": report_sha} for kind in ("candidate", "runtime", "source")}}


def ssh_command(root, row, port):
    return ["ssh", "-F", "/dev/null", "-N", "-T", "-i", str(root / "forward-key"), "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none",
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + str(root / "known_hosts"),
            "-o", "GlobalKnownHostsFile=/dev/null", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3",
            "-o", "ForwardAgent=no", "-o", "ForwardX11=no", "-o", "PermitLocalCommand=no", "-o", "RequestTTY=no", "-o", "LogLevel=ERROR",
            "-R", f"127.0.0.1:{row['forward_port']}:127.0.0.1:{port}", lease_common.SSH_USER + "@" + lease_common.HOST]


def stop(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def engine(root, row):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {"PATH": os.environ["PATH"], "HOME": str(root), "FOUNDATION_POOL_SIZE": "1",
           "FOUNDATION_MODEL_PATH": str(root / "foundation/foundation_lora_fp32.onnx"), "FOUNDATION_MODEL_HASH": contracts.FOUNDATION["foundation_lora_fp32.onnx"],
           "FOUNDATION_ENGINE_WASM_PATH": str(root / "foundation/foundation_engine.wasm"), "FOUNDATION_ENGINE_WASM_HASH": contracts.FOUNDATION["foundation_engine.wasm"]}
    process = subprocess.Popen([str(root / "adapterexec"), "--package", str(root / "THREAT_IP_REPUTATION.safetensors"), "--listen", "127.0.0.1:" + str(port)],
                               env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    identity = {"foundation_hash": contracts.FOUNDATION["foundation_lora_fp32.onnx"], "tokenizer_hash": contracts.FOUNDATION["tokenizer.json"], "adapter_hash": row["keccak"]}
    frontend = Server(("127.0.0.1", 0), identity, Backend(port))
    thread = threading.Thread(target=frontend.serve_forever, daemon=True)
    thread.start()
    return process, frontend, thread, identity


def probe_engine(frontend, identity, report):
    url = "http://127.0.0.1:" + str(frontend.server_address[1]) + "/runnable"
    def post(value):
        request = urllib.request.Request(url, data=json.dumps(value).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            raw = response.read((1 << 20) + 1)
            if len(raw) > 1 << 20:
                raise ValueError("serving engine response ceiling")
            return json.loads(raw)
    for _ in range(90):
        try:
            post({**identity, "batch": []})
            break
        except OSError:
            time.sleep(2)
    else:
        raise ValueError("serving engine readiness deadline")
    vectors = report["runtime"]["vectors"]
    for _ in range(2):
        value = post({**identity, "batch": [{"tokens": vector["tokens"]} for vector in vectors]})
        if any(value.get(key) != expected for key, expected in identity.items()) or [entry.get("score") for entry in value.get("results", [])] != [vector["score_bits"] for vector in vectors]:
            raise ValueError("actual serving executor differs from independent reference bits")
    for field in identity:
        try:
            post({**identity, field: "0" * 64, "batch": []})
        except urllib.error.HTTPError as exc:
            if exc.code != 400:
                raise ValueError("literal wrong-identity refusal differs") from exc
        else:
            raise ValueError("serving executor accepted wrong identity")


def main():
    contracts.remote_only()  # Before any download, model, key or SSH operation.
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "run", "cleanup"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--reservation", type=Path)
    parser.add_argument("--report-sha256")
    args = parser.parse_args()
    root = args.root.resolve()
    if root.parent != Path(os.environ["RUNNER_TEMP"]).resolve() or root.name != "threat-ip-serving":
        raise ValueError("owned bounded serving root required")
    if args.mode == "cleanup":
        (root / "forward-key").unlink(missing_ok=True)
        (root / "forward-key.pub").unlink(missing_ok=True)
        return
    if args.mode == "prepare":
        raw = (args.audit / "registration-ready.json").read_bytes()
        if contracts.sha(raw) != args.report_sha256:
            raise ValueError("actual audit artifact report bytes differ")
        report = ready_report(raw)
        run, _ = lease_common.public_json("https://api.github.com/repos/" + contracts.REPO + "/actions/runs/" + str(report["run_id"]))
        if run.get("head_sha") != report["commit"] or run.get("status") != "completed" or run.get("conclusion") != "success":
            raise ValueError("actual independent audit run did not pass")
        if report["serving_commit"] != os.environ["GITHUB_SHA"]:
            raise ValueError("serving source differs from the actual audited source commit")
        row = lease_row(report, json.loads(args.reservation.read_text()), contracts.sha(raw))
        package = args.audit / "THREAT_IP_REPUTATION.safetensors"
        if contracts.sha(package.read_bytes()) != row["sha256"] or package.stat().st_size != report["package_bytes"]:
            raise ValueError("exact audited package bytes required")
        if root.exists():
            raise ValueError("serving root already holds a candidate")
        root.mkdir(mode=0o700)
        shutil.copyfile(package, root / package.name)
        (root / "registration-ready.json").write_bytes(raw)
        foundation, base, kit = prepare(root)
        cfg = json.loads((base / "config.json").read_text())
        h, i = cfg["hidden_size"], cfg["intermediate_size"]
        contracts.write_json(root / "foundation.json", {"foundation_hash": contracts.FOUNDATION["foundation_lora_fp32.onnx"], "tokenizer_hash": contracts.FOUNDATION["tokenizer.json"],
                             "foundation_version": "modernbert-base-lora16-v1", "base_model": str(base), "r_max": 16, "num_layers": cfg["num_hidden_layers"], "hidden_size": h,
                             "targets": {"attn.Wqkv": [h, 3*h], "attn.Wo": [h, h], "mlp.Wi": [h, 2*i], "mlp.Wo": [i, h]}})
        subprocess.run(["go", "build", "-trimpath", "-o", str(root / "adapterexec"), "./cmd/adapterexec"], cwd=kit, check=True, timeout=600)
        live_dir = kit / "cmd/threat-owner-livecheck"
        live_dir.mkdir(exist_ok=True)
        shutil.copyfile(HERE / "livecheck.go", live_dir / "main.go")
        subprocess.run(["go", "build", "-trimpath", "-o", str(root / "livecheck"), "./cmd/threat-owner-livecheck"], cwd=kit, check=True, timeout=600)
        collect_classes(root / "live-class-proof.json")
        parsed = subprocess.run([str(root / "livecheck"), "--package", str(root / package.name), "--classes", str(root / "live-class-proof.json"), "--foundation", str(root / "foundation.json"), "--out", str(root)], capture_output=True, text=True, timeout=60)
        if parsed.returncode or json.loads(parsed.stdout)["keccak256"].removeprefix("0x") != row["keccak"]:
            raise ValueError("actual full serving parser refused audited package")
        process, frontend, thread, identity = engine(root, row)
        try:
            probe_engine(frontend, identity, report)
        finally:
            frontend.shutdown()
            frontend.server_close()
            thread.join(timeout=10)
            stop(process)
        key = root / "forward-key"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key), "-C", "threat-owner-forward"], check=True, timeout=15, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        key.chmod(0o600)
        (root / "known_hosts").write_text(lease_common.HOST_KEY)
        lease = lease_common.lease_for(row, (root / "forward-key.pub").read_text().strip(), "threat-owner", os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"])
        contracts.write_json(root / "lease.json", lease)
        contracts.write_json(root / "state.json", {"row": row, "report_sha256": args.report_sha256})
        return
    state = json.loads((root / "state.json").read_text())
    lease = json.loads((root / "lease.json").read_text())
    lease_common.validate_lease(lease, state["row"])
    if lease["run_id"] != os.environ["GITHUB_RUN_ID"] or lease["commit"] != os.environ["GITHUB_SHA"] or (root / "forward-key").stat().st_mode & 0o777 != 0o600:
        raise ValueError("serving run/key identity differs")
    raw = (root / "registration-ready.json").read_bytes()
    if contracts.sha(raw) != state["report_sha256"] or contracts.sha((root / "THREAT_IP_REPUTATION.safetensors").read_bytes()) != state["row"]["sha256"]:
        raise ValueError("served package/report changed")
    process, frontend, thread, identity = engine(root, state["row"])
    forward = None
    def interrupted(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        probe_engine(frontend, identity, ready_report(raw))
        expiry = dt.datetime.fromisoformat(lease["expires_at"])
        while dt.datetime.now(dt.timezone.utc) < expiry:
            if process.poll() is not None:
                raise ValueError("real serving executor stopped")
            if forward is None or forward.poll() is not None:
                stop(forward)
                forward = subprocess.Popen(ssh_command(root, state["row"], frontend.server_address[1]), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(10)
    finally:
        stop(forward)
        frontend.shutdown()
        frontend.server_close()
        thread.join(timeout=10)
        stop(process)
        (root / "forward-key").unlink(missing_ok=True)
        (root / "forward-key.pub").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
