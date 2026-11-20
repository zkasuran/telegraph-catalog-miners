# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Pinned runner-only executor and restricted public lease, never local serving."""
import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from data import HERE
from report import validate_ready
from runtime_check import model_remote_only

sys.path.insert(0, str(HERE / "wire"))
from common import (
    DEFAULT_OWNER,
    FOUNDATION,
    FOUNDATION_RELEASE,
    HOST,
    HOST_KEY,
    lease_for,
)
from proxy import Backend, BoundedServer, identity, validate_request, validate_response


def immutable(value):
    if not re.fullmatch(r"https://raw\.githubusercontent\.com/zkasuran/telegraph-catalog-miners/[a-f0-9]{40}/verification/owners/fact_check/[A-Za-z0-9_./-]+", value) or ".." in value.split("/"):
        raise ValueError("owned immutable public asset URL required")
    return value


def download(url, path, expected, limit):
    if not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError("full public asset hash required")
    request = urllib.request.Request(url, headers={"User-Agent": "FACT_CHECK-owner-serving/1"})
    started = time.monotonic()
    h, total = __import__("hashlib").sha256(), 0
    with urllib.request.urlopen(request, timeout=45) as response, path.open("wb") as target:
        while chunk := response.read(1 << 20):
            total += len(chunk)
            if total > limit or time.monotonic() - started > 600:
                raise ValueError("public asset byte/time ceiling")
            h.update(chunk)
            target.write(chunk)
    if h.hexdigest() != expected:
        raise ValueError("public asset hash mismatch")


def serve(args):
    model_remote_only()
    root = args.root.resolve()
    if not root.is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("runner-temporary serving only")
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "report.json"
    download(immutable(args.report_url), report_path, args.report_sha256, 256 << 10)
    report = validate_ready(json.loads(report_path.read_text()))
    if report["serving_commit"] != os.environ["GITHUB_SHA"] or report["package_sha256"] != args.package_sha256 or report["package_keccak"] != args.package_keccak:
        raise ValueError("serving source/package differs from measured audit")
    package = root / "FACT_CHECK.safetensors"
    download(immutable(args.package_url), package, args.package_sha256, 64 << 20)
    foundation = root / "foundation"
    foundation.mkdir(exist_ok=True)
    for name, expected in FOUNDATION.items():
        download(FOUNDATION_RELEASE + name, foundation / name, expected, 700 << 20)
    from run import prepare_kit
    kit = prepare_kit(root)
    executable = root / "adapterexec"
    subprocess.run(["go", "build", "-mod=readonly", "-trimpath", "-o", str(executable), "./cmd/adapterexec"], cwd=kit, check=True, timeout=600)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        backend_port = probe.getsockname()[1]
    environment = {**os.environ, "FOUNDATION_MODEL_PATH": str(foundation / "foundation_lora_fp32.onnx"),
                   "FOUNDATION_MODEL_HASH": FOUNDATION["foundation_lora_fp32.onnx"],
                   "FOUNDATION_ENGINE_WASM_PATH": str(foundation / "foundation_engine.wasm"),
                   "FOUNDATION_ENGINE_WASM_HASH": FOUNDATION["foundation_engine.wasm"], "ADAPTER_SCORE_TIMEOUT": "180s"}
    row = {"family": "FACT_CHECK", "owner": DEFAULT_OWNER, "slot": args.slot, "forward_port": args.forward_port,
           "runnable_url": args.runnable_url, "keccak": args.package_keccak, "sha256": args.package_sha256,
           "existing_registration_id": 0, **{name + "_proof": {"sha256": args.report_sha256} for name in ("candidate", "runtime", "source")}}
    if not 1 <= row["slot"] <= 6 or not 1 <= row["forward_port"] <= 65535 or not re.fullmatch(r"https://tga1\.zkasuran\.dev/_[A-Za-z0-9_-]{1,64}/runnable", row["runnable_url"]):
        raise ValueError("reserved endpoint fields invalid")
    executor = subprocess.Popen([str(executable), "--package", str(package), "--listen", f"127.0.0.1:{backend_port}"], env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    forwarding, server = None, None
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if executor.poll() is not None:
                raise ValueError("pinned executor refused package")
            try:
                with socket.create_connection(("127.0.0.1", backend_port), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError("executor startup ceiling")
        backend = Backend(backend_port, timeout=180)
        for vector in report["runtime"]["vectors"]:
            request = validate_request({"batch": [{"tokens": vector["tokens"]}], **identity(row)}, row)
            first = validate_response(backend.score(request), request, row)
            repeat = validate_response(backend.score(request), request, row)
            if first != repeat or first["results"] != [{"score": vector["score_bits"]}]:
                raise ValueError("public executor differs from actual reference vectors")
        for tokens in ([], [-1]):
            try:
                validate_request({"batch": [{"tokens": tokens}], **identity(row)}, row)
            except ValueError:
                pass
            else:
                raise ValueError("invalid token refusal failed")
        server = BoundedServer(("127.0.0.1", 0), row, backend)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        key = root / "forward-key"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key), "-C", "factcheck-owner-lease"], check=True, timeout=15)
        key.chmod(0o600)
        known = root / "known_hosts"
        known.write_text(HOST_KEY)
        public = root / "public"
        public.mkdir(exist_ok=True)
        lease = lease_for(row, key.with_suffix(".pub").read_text().strip(), "factcheck-" + os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"])
        (public / "lease.json").write_text(json.dumps(lease, indent=2) + "\n")
        command = ["ssh", "-F", "/dev/null", "-N", "-T", "-i", str(key), "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none",
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + str(known),
            "-o", "GlobalKnownHostsFile=/dev/null", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3",
            "-o", "ForwardAgent=no", "-o", "ForwardX11=no", "-o", "PermitLocalCommand=no", "-o", "RequestTTY=no",
            "-R", f"127.0.0.1:{args.forward_port}:127.0.0.1:{server.server_port}", "telegraph-verify@" + HOST]
        # The lease artifact is uploaded before activation. Wait boundedly for
        # the adapter to install this restricted public key, not another signer.
        activation = time.monotonic() + 600
        while time.monotonic() < activation:
            forwarding = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(10)
            if forwarding.poll() is None:
                break
        else:
            raise TimeoutError("restricted key was not activated within the bounded window")
        deadline = time.monotonic() + 270 * 60
        while time.monotonic() < deadline and executor.poll() is None and forwarding.poll() is None and not server.fatal.is_set():
            time.sleep(10)
        if time.monotonic() < deadline:
            raise ValueError("serving or restricted forward failed")
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        for process in (forwarding, executor):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def main():
    model_remote_only()
    if "--background" in sys.argv:
        arguments = [arg for arg in sys.argv[1:] if arg != "--background"]
        root = Path(arguments[arguments.index("--root") + 1]).resolve()
        if not root.is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
            raise ValueError("runner-temporary output only")
        root.mkdir(parents=True, exist_ok=True)
        with (root / "serving.log").open("w") as log:
            child = subprocess.Popen(["timeout", "--signal=TERM", "--kill-after=10s", "19200", sys.executable,
                                      str(HERE / "serving.py"), *arguments], stdout=log, stderr=log)
        (root / "serving-pid.json").write_text(json.dumps({"pid": child.pid}) + "\n")
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise ValueError("serving startup failed; inspect bounded runner log")
            if (root / "public/lease.json").is_file():
                print("Exact reference vectors passed; restricted public lease prepared")
                return
            time.sleep(2)
        raise TimeoutError("serving preparation exceeded startup window")
    if "--wait-root" in sys.argv:
        root = Path(sys.argv[sys.argv.index("--wait-root") + 1]).resolve()
        if not root.is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
            raise ValueError("runner-temporary output only")
        pid = json.loads((root / "serving-pid.json").read_text())["pid"]
        deadline = time.monotonic() + 285 * 60
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(10)
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    for name in ("package-url", "package-sha256", "package-keccak", "report-url", "report-sha256", "runnable-url"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--forward-port", type=int, required=True)
    signal.alarm(320 * 60)
    serve(parser.parse_args())


if __name__ == "__main__":
    main()
