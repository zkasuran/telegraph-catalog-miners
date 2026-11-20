# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only identical public executor with an ephemeral restricted forwarding lease."""
import argparse
import json
import os
import signal
import socket
import subprocess
import tarfile
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import native
from lease_common import (
    BASE_CONFIG_SHA,
    BASE_CONFIG_URL,
    HOST,
    HOST_KEY,
    SSH_USER,
    lease_for,
    strict_json,
    utc,
)
from remote import remote_only
from serving_contract import check_run, owned_url, serving_row
from source import sha, write
from token_proxy import Backend, BoundedServer, identity, input_root, validate_response

HERE = Path(__file__).resolve().parent


def public_run(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Telegraph-keyword-serving/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read((2 << 20) + 1)
        if response.status != 200 or len(raw) > 2 << 20:
            raise ValueError("public run metadata ceiling or status")
    return strict_json(raw)


def engine_env(foundation):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "FOUNDATION_MODEL_PATH": str(foundation / "foundation_lora_fp32.onnx"),
            "FOUNDATION_MODEL_HASH": native.FOUNDATION["foundation_lora_fp32.onnx"],
            "FOUNDATION_ENGINE_WASM_PATH": str(foundation / "foundation_engine.wasm"),
            "FOUNDATION_ENGINE_WASM_HASH": native.FOUNDATION["foundation_engine.wasm"]}


def prepare(root, settings):
    remote_only()
    if os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch":
        raise RuntimeError("serving is restricted to the bounded public dispatch workflow")
    temp = Path(os.environ["RUNNER_TEMP"]).resolve()
    root = root.resolve()
    if root.parent != temp or root.name != "keyword-serving":
        raise ValueError("bounded runner-owned serving root required")
    root.mkdir(mode=0o700, exist_ok=True)
    if any((root / name).exists() for name in ("forward-key", "lease.json", "serving-started.json")):
        raise ValueError("existing serving attempt must be resumed, not replaced")
    owned_url(settings["report_url"])
    report_path = root / "registration-ready.json"
    native.fetch(settings["report_url"], report_path, settings["report_sha256"], 1 << 20)
    report = strict_json(report_path.read_bytes())
    row = serving_row(settings, report, os.environ["GITHUB_SHA"])
    metadata = public_run("https://api.github.com/repos/zkasuran/telegraph-catalog-miners/actions/runs/" + row["verified_run_id"])
    check_run(metadata, row)
    write(root / "serving-started.json", {"family": row["family"], "package_sha256": row["sha256"],
          "report_sha256": settings["report_sha256"], "commit": os.environ["GITHUB_SHA"], "run_id": os.environ["GITHUB_RUN_ID"]})
    package = root / row["file"]
    native.fetch(row["url"], package, row["sha256"], 64 << 20)
    foundation = root / "foundation"
    foundation.mkdir(exist_ok=True)
    for name, digest in native.FOUNDATION.items():
        native.fetch(native.RELEASE + name, foundation / name, digest)
    config_path = root / "base-config.json"
    native.fetch(BASE_CONFIG_URL, config_path, BASE_CONFIG_SHA, 1 << 20)
    cfg = strict_json(config_path.read_bytes())
    hidden, intermediate = cfg["hidden_size"], cfg["intermediate_size"]
    if cfg["vocab_size"] != 50368:
        raise ValueError("pinned vocabulary boundary mismatch")
    spec = {"foundation_hash": native.FOUNDATION["foundation_lora_fp32.onnx"], "foundation_version": "modernbert-base-lora16-v1",
            "tokenizer_hash": native.FOUNDATION["tokenizer.json"], "r_max": 16,
            "num_layers": cfg["num_hidden_layers"], "hidden_size": hidden,
            "targets": {"attn.Wqkv": [hidden, 3 * hidden], "attn.Wo": [hidden, hidden],
                        "mlp.Wi": [hidden, 2 * intermediate], "mlp.Wo": [intermediate, hidden]}}
    write(root / "foundation.json", spec)
    kit_spec = native.SOURCES["kit"]
    archive = root / "kit.tar.gz"
    native.fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + kit_spec["revision"], archive, kit_spec["sha256"], 32 << 20)
    with tarfile.open(archive) as source:
        members = source.getmembers()
        if len(members) > 5000 or sum(m.size for m in members) > 128 << 20:
            raise ValueError("source archive extraction ceiling exceeded")
        source.extractall(root, filter="data")
    kit = root / ("telegraph-wasm-baseline-" + kit_spec["revision"])
    for name, target in (("executor", "./cmd/adapterexec"), ("checker", str(HERE / "packagecheck.go")), ("input-root", str(HERE / "input_root.go"))):
        subprocess.run(["go", "build", "-trimpath", "-o", str(root / name), target], cwd=kit, check=True, timeout=600)
    if sha((root / "executor").read_bytes()) != report["runtime"]["executor_sha256"]:
        raise ValueError("serving executor bytes differ from the actual audit")
    snapshot = {"at": datetime.now(timezone.utc).isoformat(),
                "url": "https://devnode.telegraphprotocol.com/engine/v1/intents/KEYWORD_EXTRACTION_GENERAL",
                "family_url": "https://devnode.telegraphprotocol.com/api/wasm?resolve=KEYWORD_EXTRACTION"}
    for field, url in (("child", snapshot["url"]), ("parent", snapshot["family_url"])):
        snapshot[field] = public_run(url)
    write(root / "class-snapshot.json", snapshot)
    checked = subprocess.run([str(root / "checker"), "--snapshot", str(root / "class-snapshot.json"), "--package", str(package),
                              "--foundation", str(root / "foundation.json"), "--out", str(root)], capture_output=True, text=True, check=True, timeout=60)
    manifest = strict_json(checked.stdout)
    if manifest["sha256"] != row["sha256"] or manifest["runtime_adapter_hash"] != row["keccak"] or manifest["manifest"]["verification"]["stage1"] is not True:
        raise ValueError("full parser serving identity mismatch")
    write(root / "static-manifest.json", manifest)
    return row, report, package, foundation


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def run(root, settings):
    row, report, package, foundation = prepare(root, settings)
    root = root.resolve()
    port = free_port()
    executor = None
    server = None
    forward = None
    stopped = threading.Event()
    for name in (signal.SIGTERM, signal.SIGINT):
        signal.signal(name, lambda *_: stopped.set())
    key = root / "forward-key"
    try:
        with (root / "executor.log").open("w") as log:
            executor = subprocess.Popen([str(root / "executor"), "--package", str(package), "--listen", "127.0.0.1:" + str(port)],
                                        stdout=log, stderr=log, env=engine_env(foundation))
            backend = Backend(port, timeout=180)
            request = {"batch": [{"tokens": v["tokens"]} for v in report["runtime"]["vectors"]], **identity(row)}
            request["input_root"] = input_root(request["batch"])
            primary_root = subprocess.run([str(root / "input-root")], input=json.dumps([v["tokens"] for v in report["runtime"]["vectors"]]),
                                          capture_output=True, text=True, timeout=15, check=True).stdout.strip()
            if primary_root != request["input_root"]:
                raise ValueError("primary input-root wire mismatch")
            deadline = time.monotonic() + 180
            while True:
                if executor.poll() is not None or time.monotonic() > deadline or stopped.is_set():
                    raise RuntimeError("reference engine serving readiness failed")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    time.sleep(1)
            for _ in range(2):
                response = validate_response(backend.score(request), request, row)
                if [v["score"] for v in response["results"]] != [v["score_bits"] for v in report["runtime"]["vectors"]]:
                    raise ValueError("actual serving reference bits differ from the frozen audit")
            server = BoundedServer(("127.0.0.1", 0), row, backend)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key), "-C", "keyword-public-runner"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=True)
            key.chmod(0o600)
            (root / "known_hosts").write_text(HOST_KEY)
            lease = lease_for(row, key.with_suffix(".pub").read_text().strip(), "keyword-" + os.environ["GITHUB_RUN_ID"],
                              os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"])
            write(root / "lease.json", lease)
            expires = utc(lease["expires_at"]).timestamp()
            print("PUBLIC_RESTRICTED_LEASE_READY KEYWORD_EXTRACTION", flush=True)
            command = ["ssh", "-F", "/dev/null", "-N", "-T", "-i", str(key), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
                       "-o", "IdentityAgent=none", "-o", "ForwardAgent=no", "-o", "StrictHostKeyChecking=yes", "-o", "GlobalKnownHostsFile=/dev/null",
                       "-o", "UserKnownHostsFile=" + str(root / "known_hosts"), "-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=10",
                       "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3", "-o", "RequestTTY=no", "-o", "LogLevel=ERROR",
                       "-R", f"127.0.0.1:{row['forward_port']}:127.0.0.1:{server.server_port}", SSH_USER + "@" + HOST]
            with (root / "forward.log").open("w") as forwarding_log:
                activation_deadline = time.monotonic() + 900
                connected = False
                while not stopped.is_set() and time.time() < expires:
                    if server.fatal.is_set() or executor.poll() is not None:
                        raise RuntimeError("actual serving engine failed; lease cannot continue")
                    if forward is None or forward.poll() is not None:
                        if connected or time.monotonic() > activation_deadline:
                            raise RuntimeError("restricted forwarding session failed")
                        forward = subprocess.Popen(command, stdout=forwarding_log, stderr=forwarding_log,
                                                   env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")})
                        time.sleep(5)
                        if forward.poll() is None:
                            connected = True
                    stopped.wait(1)
    finally:
        for process in (forward, executor):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        if server is not None:
            server.shutdown()
            server.server_close()
        key.unlink(missing_ok=True)
        key.with_suffix(".pub").unlink(missing_ok=True)


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    for name in ("package-url", "report-url", "report-sha256", "runnable-url"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--forward-port", type=int, required=True)
    args = parser.parse_args()
    settings = {k: v for k, v in vars(args).items() if k != "root"}
    run(args.root, settings)


if __name__ == "__main__":
    main()
