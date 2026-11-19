# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Remote-only serving of the exact audited delta through a restricted public lease."""
import argparse
import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from data import HERE, digest, write_json
from ready import verify_ready
from reference import BASE_COMMIT, BASE_FILES, RELEASE, fetch, prepare_kit
from remote import remote_only
from runtime import HASHES, compile_tools, server, validate_package
from wire_common import DEFAULT_OWNER, HOST_KEY, WINDOW, immutable_url, lease_for, strict_json, utc
from wire_proxy import Backend, BoundedServer, identity, input_root
from wire_session import ssh_command, terminate


def download_public(url, target, sha, ceiling):
    immutable_url(url)
    prefix = "https://raw.githubusercontent.com/zkasuran/telegraph-catalog-miners/"
    _, _, suffix = url[len(prefix):].partition("/")
    if not suffix.startswith("verification/owners/email_security/"):
        raise ValueError("package/report URL escapes owned public namespace")
    fetch(url, target, sha, ceiling)


def row_from_report(report, report_sha, slot, port, runnable, existing=0):
    verify_ready(report)
    if type(slot) is not int or not 1 <= slot <= 6 or port != 48600 + slot:
        raise ValueError("reserved free endpoint slot and port required")
    import re
    if not re.fullmatch(r"https://tga1\.zkasuran\.dev/_[A-Za-z0-9_-]{1,64}/runnable", runnable):
        raise ValueError("existing additive proxy URL required")
    if type(existing) is not int or not 0 <= existing < 1 << 63:
        raise ValueError("invalid exact existing registration identity")
    return {"family": "EMAIL_SECURITY", "child": "EMAIL_SECURITY_GENERAL", "owner": DEFAULT_OWNER,
            "class": "judgment", "slot": slot, "forward_port": port, "runnable_url": runnable,
            "keccak": report["package_keccak"], "sha256": report["package_sha256"], "existing_registration_id": existing,
            "candidate_proof": {"sha256": report_sha}, "runtime_proof": {"sha256": report_sha}, "source_proof": {"sha256": report_sha}}


def check_vector_reply(reply, report, vectors):
    wanted = {"foundation_hash": report["foundation_sha256"], "adapter_hash": report["package_keccak"],
              "tokenizer_hash": report["tokenizer_sha256"]}
    if not isinstance(reply, dict) or any(reply.get(k) != v for k, v in wanted.items()):
        raise ValueError("literal bare public executor identity mismatch")
    if reply.get("results") != [{"score": vector["score_bits"]} for vector in vectors]:
        raise ValueError("public executor changed audited score bits")
    batch = [{"tokens": vector["tokens"]} for vector in vectors]
    if reply.get("input_root") != input_root(batch):
        raise ValueError("public executor literal input root mismatch")


def prepare(args):
    remote_only()
    root = args.root
    if root.exists():
        raise ValueError("fresh serving run directory required")
    root.mkdir(parents=True)
    download_public(args.report_url, root / "registration-ready.json", args.report_sha256, 1 << 20)
    report = verify_ready(strict_json((root / "registration-ready.json").read_bytes()))
    if report["serving_commit"] != os.environ["GITHUB_SHA"]:
        raise ValueError("serving source commit differs from audited executor source")
    if report["source"]["source_manifest_sha256"] != digest((HERE / "source-manifest.json").read_bytes()):
        raise ValueError("serving source manifest differs from audited source")
    row = row_from_report(report, args.report_sha256, args.slot, args.forward_port, args.runnable_url, args.registration_id)
    download_public(args.package_url, root / "EMAIL_SECURITY.safetensors", report["package_sha256"], 64 << 20)
    foundation = root / "foundation"
    foundation.mkdir()
    for name, sha in HASHES.items():
        fetch(RELEASE + name, foundation / name, sha, 700 << 20)
    config_path = root / "base-config.json"
    fetch("https://huggingface.co/answerdotai/ModernBERT-base/resolve/" + BASE_COMMIT + "/config.json",
          config_path, BASE_FILES["config.json"], 1 << 20)
    cfg = json.loads(config_path.read_bytes())
    h, i = cfg["hidden_size"], cfg["intermediate_size"]
    write_json(root / "foundation.json", {"foundation_hash": HASHES["foundation_lora_fp32.onnx"],
               "foundation_version": "modernbert-base-lora16-v1", "tokenizer_hash": HASHES["tokenizer.json"],
               "r_max": 16, "hidden_size": h, "num_layers": cfg["num_hidden_layers"],
               "targets": {"attn.Wqkv": [h, 3*h], "attn.Wo": [h, h], "mlp.Wi": [h, 2*i], "mlp.Wo": [i, h]}})
    kit = prepare_kit(root)
    compile_tools(root, kit)
    manifest = validate_package(root)
    if manifest["keccak256"].removeprefix("0x") != report["package_keccak"]:
        raise ValueError("served package differs from audited package")
    vectors = report["runtime"]["vectors"]
    with server(root, manifest, "serving-preflight", time.monotonic() + 1800) as score:
        actual = [bit for offset in range(0, len(vectors), 4) for bit in score([v["tokens"] for v in vectors[offset:offset+4]])]
    if actual != [vector["score_bits"] for vector in vectors]:
        raise ValueError("fresh serving engine changed audited fixed vectors")
    # An ephemeral forwarding key is created only on this public runner. The
    # agent receives no private key and uploads only the restricted public lease.
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(root / "forward-key"),
                    "-C", "email-security-forward"], check=True, timeout=15)
    (root / "forward-key").chmod(0o600)
    (root / "known_hosts").write_text(HOST_KEY)
    key = (root / "forward-key.pub").read_text().strip()
    lease = lease_for(row, key, "email-security-" + os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"])
    write_json(root / "lease.json", lease)
    write_json(root / "serving-state.json", {"row": row, "manifest": manifest, "report_sha256": args.report_sha256,
               "created": lease["created_at"], "expires": lease["expires_at"]})
    print(json.dumps({"status": "PUBLIC_RESTRICTED_LEASE_READY", "lease": "lease.json", "fixed_vectors": len(vectors),
                      "package_sha256": report["package_sha256"], "run_id": int(os.environ["GITHUB_RUN_ID"])}))


def run(args):
    remote_only()
    root = args.root
    state = strict_json((root / "serving-state.json").read_bytes())
    report = verify_ready(strict_json((root / "registration-ready.json").read_bytes()))
    if digest((root / "registration-ready.json").read_bytes()) != state["report_sha256"]:
        raise ValueError("serving report changed after preparation")
    row = state["row"]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        engine_port = probe.getsockname()[1]
    environment = {**os.environ, "FOUNDATION_MODEL_PATH": str(root / "foundation/foundation_lora_fp32.onnx"),
                   "FOUNDATION_MODEL_HASH": HASHES["foundation_lora_fp32.onnx"],
                   "FOUNDATION_ENGINE_WASM_PATH": str(root / "foundation/foundation_engine.wasm"),
                   "FOUNDATION_ENGINE_WASM_HASH": HASHES["foundation_engine.wasm"], "ADAPTER_SCORE_TIMEOUT": "180s"}
    process = subprocess.Popen([str(root / "adapterexec"), "--package", str(root / "EMAIL_SECURITY.safetensors"),
                                "--listen", f"127.0.0.1:{engine_port}"], env=environment, start_new_session=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proxy, forward = None, None
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("serving reference executor failed to load")
            try:
                with socket.create_connection(("127.0.0.1", engine_port), timeout=1):
                    break
            except OSError:
                time.sleep(.5)
        else:
            raise TimeoutError("serving reference executor readiness ceiling")
        backend = Backend(engine_port, timeout=180)
        proxy = BoundedServer(("127.0.0.1", 0), row, backend)
        thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        thread.start()
        vectors = report["runtime"]["vectors"]
        batch = [{"tokens": vector["tokens"]} for vector in vectors]
        request = urllib.request.Request(f"http://127.0.0.1:{proxy.server_port}/runnable",
                                         data=json.dumps({"batch": batch, **identity(row), "input_root": input_root(batch)}).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            check_vector_reply(strict_json(response.read((1 << 20) + 1)), report, vectors)
        for tokens in ([], [-1]):
            request = urllib.request.Request(f"http://127.0.0.1:{proxy.server_port}/runnable",
                                             data=json.dumps({"batch": [{"tokens": tokens}], **identity(row)}).encode(),
                                             headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(request, timeout=15).close()
            except urllib.error.HTTPError as error:
                if error.code != 400 or strict_json(error.read()) != {"error": "request refused"}:
                    raise ValueError("public token refusal contract changed") from error
            else:
                raise ValueError("public proxy accepted invalid tokens")
        write_json(root / "serving-smoke.json", {"status": "PASS", "vectors": len(vectors), "invalid_tokens_refused": 2,
                   "package_sha256": report["package_sha256"], "report_sha256": state["report_sha256"],
                   "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"]})
        end = min(utc(state["expires"]).timestamp(), time.time() + WINDOW)
        while time.time() < end:
            if process.poll() is not None or proxy.fatal.is_set():
                raise RuntimeError("exact serving executor or bounded proxy failed")
            if forward is None or forward.poll() is not None:
                # Activation installs this restricted public key. Until then
                # reconnecting the same forward has no signing or registry write.
                forward = subprocess.Popen(ssh_command(root, row, proxy.server_port), start_new_session=True,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(15)
    finally:
        terminate(forward)
        if proxy is not None:
            proxy.shutdown()
            proxy.server_close()
        terminate(process)


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--package-url")
    parser.add_argument("--report-url")
    parser.add_argument("--report-sha256")
    parser.add_argument("--slot", type=int)
    parser.add_argument("--forward-port", type=int)
    parser.add_argument("--runnable-url")
    parser.add_argument("--registration-id", type=int, default=0)
    args = parser.parse_args()
    runner = Path(os.environ["RUNNER_TEMP"]).resolve()
    if not args.root.is_absolute() or not args.root.resolve().is_relative_to(runner) or args.root.resolve() == runner:
        raise ValueError("serving state stays inside this runner's temporary directory")
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError("serving workflow ceiling")))
    return prepare(args) if args.mode == "prepare" else run(args)


if __name__ == "__main__":
    main()
