# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Runner-only reference executor and ephemeral restricted public lease."""
import argparse
import json
import os
import signal
import socket
import subprocess
import threading
import time
from contextlib import ExitStack
from pathlib import Path

from common import DEFAULT_OWNER, HOST, HOST_KEY, immutable_url, lease_for, strict_json
from contracts import FAMILY, ready_report, sha, write
from proxy import Backend, BoundedServer
from remote import fetch, parse_package, prepare, remote_only


def prepare_serving(root, settings):
    remote_only()
    root.mkdir(parents=True, exist_ok=True)
    expected = {"package_url", "package_sha256", "package_keccak", "report_url", "report_sha256", "slot", "forward_port", "runnable_url"}
    if set(settings) != expected or type(settings["slot"]) is not int or not 1 <= settings["slot"] <= 6 or settings["forward_port"] != 48600 + settings["slot"]:
        raise ValueError("automatic free slot and exact serving settings required")
    immutable_url(settings["package_url"])
    immutable_url(settings["report_url"])
    report_path = root / "registration-ready.json"
    fetch(settings["report_url"], report_path, settings["report_sha256"], 1 << 20)
    report = ready_report(strict_json(report_path.read_bytes()))
    if (report["serving_commit"] != os.environ["GITHUB_SHA"] or report["package_keccak"] != settings["package_keccak"] or
            report["package_sha256"] != settings["package_sha256"]):
        raise ValueError("serving source or exact audited package mismatch")
    prepare(root, training=False)
    package = root / "WEB_SEARCH.safetensors"
    fetch(settings["package_url"], package, settings["package_sha256"], 64 << 20)
    manifest = parse_package(root, package)
    if manifest["keccak256"] != "0x" + settings["package_keccak"]:
        raise ValueError("audited package keccak mismatch")
    key = root / "forward-key"
    if key.exists():
        raise ValueError("serving lease already exists")
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key), "-C", "web-search-lease"],
                   check=True, timeout=15, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    key.chmod(0o600)
    row = {"family": FAMILY, "owner": DEFAULT_OWNER, "slot": settings["slot"], "forward_port": settings["forward_port"],
           "runnable_url": settings["runnable_url"], "keccak": settings["package_keccak"], "sha256": settings["package_sha256"],
           "existing_registration_id": 0, **{name + "_proof": {"sha256": settings["report_sha256"]} for name in ("candidate", "runtime", "source")}}
    lease = lease_for(row, (root / "forward-key.pub").read_text().strip(), "web-search-" + os.environ["GITHUB_RUN_ID"],
                      os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"])
    write(root / "lease.json", lease)
    write(root / "serving-state.json", row)
    (root / "known_hosts").write_text(HOST_KEY)


def stop_process(process):
    """Release one owned child with bounded waits, including an exited child."""
    if process is None:
        return
    if process.poll() is not None:
        process.wait(timeout=5)
        return
    try:
        process.terminate()
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def serve(root):
    remote_only()
    from common import FOUNDATION
    # Register each resource as it becomes owned. Every callback runs even when
    # another cleanup raises. Never call shutdown on an unstarted server loop.
    with ExitStack() as resources:
        for name in ("forward-key.pub", "forward-key"):
            resources.callback((root / name).unlink, missing_ok=True)
        row = strict_json((root / "serving-state.json").read_bytes())
        if sha((root / "WEB_SEARCH.safetensors").read_bytes()) != row["sha256"]:
            raise ValueError("serving package bytes changed")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = {"PATH": os.environ["PATH"], "HOME": str(root), "FOUNDATION_POOL_SIZE": "1",
               "FOUNDATION_MODEL_PATH": str(root / "foundation/foundation_lora_fp32.onnx"),
               "FOUNDATION_MODEL_HASH": FOUNDATION["foundation_lora_fp32.onnx"],
               "FOUNDATION_ENGINE_WASM_PATH": str(root / "foundation/foundation_engine.wasm"),
               "FOUNDATION_ENGINE_WASM_HASH": FOUNDATION["foundation_engine.wasm"]}
        executor = subprocess.Popen([str(root / "adapterexec"), "--package", str(root / "WEB_SEARCH.safetensors"), "--listen", f"127.0.0.1:{port}"],
                                    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        resources.callback(stop_process, executor)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if executor.poll() is not None:
                raise RuntimeError("serving reference engine failed")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(.5)
        else:
            raise TimeoutError("serving reference engine readiness ceiling")
        proxy = BoundedServer(("127.0.0.1", 0), row, Backend(port, timeout=180))
        resources.callback(proxy.server_close)
        thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        thread.start()
        resources.callback(thread.join, timeout=5)
        resources.callback(proxy.shutdown)
        forward = None
        resources.callback(lambda: stop_process(forward))
        command = ["ssh", "-F", "/dev/null", "-N", "-T", "-i", str(root / "forward-key"),
                   "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                   "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + str(root / "known_hosts"),
                   "-o", "GlobalKnownHostsFile=/dev/null", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=20",
                   "-o", "ServerAliveCountMax=3", "-o", "ForwardAgent=no", "-o", "ForwardX11=no", "-o", "PermitLocalCommand=no",
                   "-o", "RequestTTY=no", "-o", "LogLevel=ERROR", "-R", f"127.0.0.1:{row['forward_port']}:127.0.0.1:{proxy.server_port}",
                   "telegraph-verify@" + HOST]
        expires = time.monotonic() + 290 * 60
        while time.monotonic() < expires:
            if executor.poll() is not None or proxy.fatal.is_set():
                raise RuntimeError("audited executor stopped")
            if forward is None or forward.poll() is not None:
                if forward is not None:
                    forward.wait(timeout=5)
                forward = subprocess.Popen(command, env={"PATH": os.environ["PATH"], "HOME": str(root)},
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(5)


if __name__ == "__main__":
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--settings", default="{}")
    args = parser.parse_args()
    def stop(signum, frame):
        raise TimeoutError("bounded serving window ended")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(300 * 60)
    if args.action == "prepare":
        prepare_serving(args.root.resolve(), json.loads(args.settings))
    else:
        serve(args.root.resolve())
