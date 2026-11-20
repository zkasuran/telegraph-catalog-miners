# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Exact immutable package serving through the existing restricted free proxy."""
import argparse
import importlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

from contracts import CHILD, FAMILY, FOUNDATION, HERE, REPO, remote_only, sha, validate_report, write
from remote import capture_class, compile_tools, support
from transport import RuntimeBridge, live_controls, proxy_module


def immutable(url):
    value = urllib.parse.urlsplit(url)
    prefix = "https://raw.githubusercontent.com/" + REPO + "/"
    if not url.startswith(prefix) or value.query or value.fragment or value.username or value.password:
        raise ValueError("public immutable raw repository URL required")
    path = url[len(prefix):]
    if not re.fullmatch(r"[0-9a-f]{40}/[A-Za-z0-9_./-]+", path) or ".." in path.split("/"):
        raise ValueError("exact public commit required")
    return url


def prepare(root):
    remote_only()
    common = registration_common()
    _, original, _ = support()["core"].load_shared()
    if root.exists():
        raise ValueError("fresh serving directory required")
    (root / "evidence").mkdir(parents=True)
    original.prepare(root)
    compile_tools(root)
    package, report_file = root / (FAMILY + ".safetensors"), root / "registration-ready.json"
    data = support()["core"].load_shared()[0]
    data.fetch(immutable(os.environ["PACKAGE_URL"]), package, os.environ["PACKAGE_SHA256"], 64 << 20)
    data.fetch(immutable(os.environ["REPORT_URL"]), report_file, os.environ["REPORT_SHA256"], 2 << 20)
    report = json.loads(report_file.read_bytes())
    validate_report(report)
    if report["package_sha256"] != sha(package.read_bytes()) or report["package_keccak"] != os.environ["PACKAGE_KECCAK"] or report["serving_commit"] != os.environ["GITHUB_SHA"]:
        raise ValueError("serving package/report/source identity mismatch")
    capture_class(root)
    slot, port = int(os.environ["SLOT"]), int(os.environ["FORWARD_PORT"])
    url = os.environ["RUNNABLE_URL"]
    if not 1 <= slot <= 100 or not 1024 <= port <= 65535 or not re.fullmatch(r"https://tga1\.zkasuran\.dev/_[A-Za-z0-9_-]{1,64}/runnable", url):
        raise ValueError("reserved existing-proxy route required")
    row = {"family": FAMILY, "child": CHILD, "owner": common.DEFAULT_OWNER, "slot": slot, "forward_port": port,
           "runnable_url": url, "keccak": report["package_keccak"], "sha256": report["package_sha256"], "existing_registration_id": 0,
           **{key + "_proof": {"sha256": os.environ["REPORT_SHA256"]} for key in ("candidate", "runtime", "source")}}
    key = root / "forward-key"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key), "-C", "owner-cm-" + os.environ["GITHUB_RUN_ID"]], check=True, timeout=15)
    key.chmod(0o600)
    (root / "known_hosts").write_text(common.HOST_KEY)
    lease = common.lease_for(row, key.with_suffix(".pub").read_text().strip(), "cm-" + os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"])
    write(root / "lease.json", lease)
    write(root / "serve-state.json", {"row": row, "report_sha256": os.environ["REPORT_SHA256"]})


def registration_common():
    sys.path.insert(0, str(HERE / "support/registration"))
    return importlib.import_module("common")


def hold(root):
    remote_only()
    common = registration_common()
    state = json.loads((root / "serve-state.json").read_bytes())
    row, package = state["row"], root / (FAMILY + ".safetensors")
    report_bytes = (root / "registration-ready.json").read_bytes()
    if sha(report_bytes) != state["report_sha256"] or sha(package.read_bytes()) != row["sha256"]:
        raise ValueError("serving bytes changed after lease preparation")
    report = json.loads(report_bytes)
    validate_report(report)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    environment = {"PATH": os.environ["PATH"], "HOME": str(root), "FOUNDATION_POOL_SIZE": "1",
                   "FOUNDATION_MODEL_PATH": str(root / "foundation/foundation_lora_fp32.onnx"), "FOUNDATION_MODEL_HASH": FOUNDATION["foundation_lora_fp32.onnx"],
                   "FOUNDATION_ENGINE_WASM_PATH": str(root / "foundation/foundation_engine.wasm"), "FOUNDATION_ENGINE_WASM_HASH": FOUNDATION["foundation_engine.wasm"]}
    executor, forward, server = None, None, None
    deadline = time.monotonic() + 5 * 3600
    with (root / "serve.log").open("wb") as log:
        try:
            executor = subprocess.Popen([str(root / "adapterexec"), "--package", str(package), "--listen", "127.0.0.1:" + str(port)], env=environment, stdout=log, stderr=log)
            ready = time.monotonic() + 180
            while time.monotonic() < ready:
                if executor.poll() is not None:
                    raise RuntimeError("exact reference executor exited")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError("serving reference-engine readiness ceiling")
            bridge = RuntimeBridge(proxy_module().Backend(port), row["keccak"])
            vectors, _ = live_controls(bridge, row["keccak"])
            if vectors != report["runtime"]["vectors"]:
                raise ValueError("served package changed actual reference-vector score bits")
            server = proxy_module().BoundedServer(("127.0.0.1", 0), row, bridge)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            proxy_port = server.server_address[1]
            command = ["ssh", "-F", "/dev/null", "-N", "-T", "-i", str(root / "forward-key"),
                       "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none", "-o", "BatchMode=yes",
                       "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + str(root / "known_hosts"),
                       "-o", "GlobalKnownHostsFile=/dev/null", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=20",
                       "-o", "ServerAliveCountMax=3", "-o", "ForwardAgent=no", "-o", "ForwardX11=no", "-o", "PermitLocalCommand=no",
                       "-o", "RequestTTY=no", "-o", "LogLevel=ERROR", "-R", "127.0.0.1:" + str(row["forward_port"]) + ":127.0.0.1:" + str(proxy_port), "telegraph-verify@" + common.HOST]
            while time.monotonic() < deadline:
                if executor.poll() is not None or server.fatal.is_set():
                    raise RuntimeError("serving reference engine or bounded wire failed")
                if forward is None or forward.poll() is not None:
                    forward = subprocess.Popen(command, stdout=log, stderr=log)
                time.sleep(5)
        finally:
            if server:
                server.shutdown()
                server.server_close()
            for process in (forward, executor):
                if process and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
            (root / "forward-key").unlink(missing_ok=True)


if __name__ == "__main__":
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "hold"))
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if root.parent != Path(os.environ["RUNNER_TEMP"]).resolve():
        raise ValueError("runner-owned serving directory required")
    {"prepare": prepare, "hold": hold}[args.phase](root)
