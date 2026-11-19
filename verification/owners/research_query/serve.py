# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only serving of exact audited bytes with a restricted public lease."""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from common import HERE, SHARED, capture_class, compile_checker, module, remote_only, source_kit
from report import FOUNDATION, ENGINE, TOKENIZER, literal_reply, validate_ready
from source import sha, write_json


def lease_modules():
    lease = module("owner_lease_contract", HERE / "serving/common.py")
    previous = sys.modules.get("common")
    sys.modules["common"] = lease
    proxy = module("owner_bounded_proxy", HERE / "serving/proxy.py")
    if previous is None:
        sys.modules.pop("common", None)
    else:
        sys.modules["common"] = previous
    return lease, proxy


def prepare(root):
    remote_only()
    root.mkdir(parents=True, exist_ok=True)
    lease, _ = lease_modules()
    package_url, report_url = os.environ["PACKAGE_URL"], os.environ["REPORT_URL"]
    lease.immutable_url(package_url)
    lease.immutable_url(report_url)
    report_sha = os.environ["REPORT_SHA256"]
    lease.fetch_hash(report_url, root / "registration-ready.json", report_sha, 128 << 10)
    report = lease.strict_json((root / "registration-ready.json").read_bytes())
    validate_ready(report)
    if report["serving_commit"] != os.environ["GITHUB_SHA"]:
        raise ValueError("serving source commit differs from audited executor")
    row = {"family": "RESEARCH_QUERY", "owner": lease.DEFAULT_OWNER, "slot": int(os.environ["SLOT"]),
           "forward_port": int(os.environ["FORWARD_PORT"]), "runnable_url": os.environ["RUNNABLE_URL"],
           "keccak": report["package_keccak"], "sha256": report["package_sha256"], "existing_registration_id": 0,
           "candidate_proof": {"sha256": report_sha}, "runtime_proof": {"sha256": report_sha}, "source_proof": {"sha256": report_sha}}
    if not 1 <= row["slot"] <= 6 or row["forward_port"] != 48600 + row["slot"] or not __import__("re").fullmatch(r"https://tga1\.zkasuran\.dev/_[A-Za-z0-9_-]{1,64}/runnable", row["runnable_url"]):
        raise ValueError("free endpoint reservation does not match restricted lease")
    lease.fetch_hash(package_url, root / "RESEARCH_QUERY.safetensors", row["sha256"], 64 << 20)
    if (root / "RESEARCH_QUERY.safetensors").stat().st_size != report["package_bytes"]:
        raise ValueError("audited package size changed")
    foundation = root / "foundation"
    foundation.mkdir(exist_ok=True)
    for name, expected in lease.FOUNDATION.items():
        lease.fetch_hash(lease.FOUNDATION_RELEASE + name, foundation / name, expected, 650 << 20)
    kit = source_kit(root)
    compile_checker(root, kit, "go")
    layout = root / "foundation.json"
    fetcher = module("owner_serving_fetch", SHARED / "training/data.py")
    fetcher.fetch(lease.FOUNDATION_RELEASE + "foundation.json", layout,
                  "f7de64b999acd3f735118fbe5999abf251fdf188f9797e3a4e36c54ee9e4dbbe", 64 << 10)
    snapshot = capture_class(root)
    checked = subprocess.run([str(root / "packagecheck"), "--intent", row["family"], "--child", "RESEARCH_QUERY_GENERAL",
                              "--class-snapshot", str(root / "class-snapshot.json"), "--snapshot-sha256", snapshot,
                              "--package", str(root / "RESEARCH_QUERY.safetensors"), "--sha256", row["sha256"],
                              "--foundation", str(layout), "--foundation-sha256", sha(layout.read_bytes()), "--out", str(root)],
                             capture_output=True, check=True, timeout=60)
    manifest = json.loads(checked.stdout)
    if manifest["keccak256"].removeprefix("0x") != row["keccak"]:
        raise ValueError("full parser package Keccak mismatch")
    subprocess.run(["go", "build", "-mod=readonly", "-trimpath", "-o", str(root / "adapterexec"), "./cmd/adapterexec"], cwd=kit, check=True, timeout=600)
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "owner-research-query-forward", "-f", str(root / "forward-key")], check=True, timeout=15)
    public = (root / "forward-key.pub").read_text().strip()
    value = lease.lease_for(row, public, "owner-research-query", os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"])
    write_json(root / "lease.json", value)
    write_json(root / "state.json", {"row": row, "report": report, "manifest": manifest})
    (root / "known_hosts").write_text(lease.HOST_KEY)


def run(root):
    remote_only()
    lease, proxy = lease_modules()
    state = json.loads((root / "state.json").read_text())
    row, report, manifest = state["row"], state["report"], state["manifest"]
    value = json.loads((root / "lease.json").read_text())
    lease.validate_lease(value, row)
    for name, expected in {"foundation_lora_fp32.onnx": FOUNDATION, "foundation_engine.wasm": ENGINE, "tokenizer.json": TOKENIZER}.items():
        digest = hashlib.sha256()
        with (root / "foundation" / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError("serving runtime bytes changed")
    if sha((root / "RESEARCH_QUERY.safetensors").read_bytes()) != row["sha256"]:
        raise ValueError("serving package bytes changed")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {"PATH": os.environ["PATH"], "HOME": str(root), "FOUNDATION_POOL_SIZE": "1",
           "FOUNDATION_MODEL_PATH": str(root / "foundation/foundation_lora_fp32.onnx"), "FOUNDATION_MODEL_HASH": FOUNDATION,
           "FOUNDATION_ENGINE_WASM_PATH": str(root / "foundation/foundation_engine.wasm"), "FOUNDATION_ENGINE_WASM_HASH": ENGINE}
    executor = subprocess.Popen([str(root / "adapterexec"), "--package", str(root / "RESEARCH_QUERY.safetensors"), "--listen", f"127.0.0.1:{port}"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    forward = None
    server = None
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if executor.poll() is not None:
                raise ValueError("exact public executor exited")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(.5)
        else:
            raise TimeoutError("executor readiness limit")
        underlying = proxy.Backend(port)
        class LiteralBackend:
            def score(self, request):
                result = underlying.score(request)
                literal_reply(result, len(request["batch"]), manifest)
                return result
        backend = LiteralBackend()
        for vector in report["runtime"]["vectors"]:
            request = {"batch": [{"tokens": vector["tokens"]}], **proxy.identity(row)}
            for _ in range(2):
                result = literal_reply(backend.score(request), 1, manifest)
                if result != [vector["score_bits"]]:
                    raise ValueError("serving bits do not reproduce audited reference vectors")
        server = proxy.BoundedServer(("127.0.0.1", 0), row, backend)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        arguments = ["ssh", "-F", "/dev/null", "-N", "-T", "-i", str(root / "forward-key"), "-o", "IdentitiesOnly=yes",
                     "-o", "IdentityAgent=none", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes",
                     "-o", "UserKnownHostsFile=" + str(root / "known_hosts"), "-o", "GlobalKnownHostsFile=/dev/null",
                     "-o", "ExitOnForwardFailure=yes", "-o", "ForwardAgent=no", "-o", "RequestTTY=no", "-o", "ServerAliveInterval=20",
                     "-o", "ServerAliveCountMax=3", "-R", f"127.0.0.1:{row['forward_port']}:127.0.0.1:{server.server_port}", lease.SSH_USER + "@" + lease.HOST]
        until = time.monotonic() + 600
        while time.monotonic() < until:
            forward = subprocess.Popen(arguments, env={"PATH": os.environ["PATH"], "HOME": str(root)}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)
            if forward.poll() is None:
                break
            forward.wait(timeout=10)
            time.sleep(5)
        else:
            raise TimeoutError("restricted forwarding authorization did not arrive")
        print("PUBLIC_EXECUTOR_READY RESEARCH_QUERY", flush=True)
        while lease.now() < lease.utc(value["expires_at"]):
            if executor.poll() is not None or forward.poll() is not None or server.fatal.is_set():
                raise ValueError("serving process or restricted forwarding failed")
            time.sleep(5)
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        for child in (forward, executor):
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
        for name in ("forward-key", "forward-key.pub"):
            (root / name).unlink(missing_ok=True)


if __name__ == "__main__":
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("serving state must stay inside runner temporary storage")
    prepare(root) if args.prepare else run(root)
