# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Verified public executor and restricted ephemeral forwarding lease, remote only."""
import argparse
import importlib.util
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bootstrap import MODEL_FILES, RELEASE, compile_tools, dump, fetch, file_sha, prepare_kit
from engine import server
from report import ready
from scorer import remote_only
from source import HERE, sha
from wire import FOUNDATION, TOKENIZER, input_root, request, response, strict_json


def immutable_url(url):
    if not isinstance(url, str) or not re.fullmatch(r"https://raw\.githubusercontent\.com/zkasuran/telegraph-catalog-miners/[0-9a-f]{40}/verification/owners/academic_search/[A-Za-z0-9_./-]+", url) or ".." in url.split("/"):
        raise ValueError("immutable owned package/proof URL required")
    return url


def validate_reservation(slot, port, url):
    if type(slot) is not int or not 1 <= slot <= 6 or type(port) is not int or port != 48600 + slot:
        raise ValueError("verified free slot/port required")
    if not re.fullmatch(r"https://tga1\.zkasuran\.dev/_[A-Za-z0-9_-]{1,64}/runnable", url):
        raise ValueError("existing owned proxy route required")


def public_request(value, keccak):
    if not isinstance(value, dict) or set(value) - {"batch", "foundation_hash", "adapter_hash", "tokenizer_hash", "input_root"}:
        raise ValueError("token-only wire fields required")
    if (value.get("foundation_hash"), value.get("adapter_hash"), value.get("tokenizer_hash")) != (FOUNDATION, keccak, TOKENIZER):
        raise ValueError("literal bare runtime identity required")
    batch = value.get("batch")
    if not isinstance(batch, list) or any(not isinstance(v, dict) or set(v) != {"tokens"} for v in batch):
        raise ValueError("batch of token-only objects required")
    vectors = [v["tokens"] for v in batch]
    if not 1 <= len(vectors) <= 100 or any(not isinstance(v, list) or not 1 <= len(v) <= 512 or any(type(t) is not int or not 0 <= t < 50368 for t in v) for v in vectors):
        raise ValueError("bounded token arrays required")
    root = input_root(vectors)
    # The literal public engine accepts up to 512 IDs and itself applies the
    # exported 256-token cap. Never pre-truncate inputs before their root.
    sent = {"batch": batch, "foundation_hash": FOUNDATION, "adapter_hash": keccak, "tokenizer_hash": TOKENIZER, "input_root": root}
    if "input_root" in value and value["input_root"] != root:
        raise ValueError("literal input root mismatch")
    return sent


def lease_module():
    path = HERE.parents[1] / "registration/common.py"
    bindings = json.loads((HERE / "dependency-bindings.json").read_bytes())
    if file_sha(path) != bindings["registration/common.py"]:
        raise ValueError("reviewed existing lease contract source changed")
    spec = importlib.util.spec_from_file_location("academic_existing_lease_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def backend(root):
    remote_only()
    plan = json.loads((root / "serve-plan.json").read_bytes())
    semaphore = threading.BoundedSemaphore(4)
    with server(root, root / "ACADEMIC_SEARCH.safetensors", "serving", 5 * 3600, request_timeout=600) as (score, post, identity):
        if identity["sha256"] != plan["package_sha256"] or identity["keccak"] != plan["package_keccak"]:
            raise ValueError("actual serving package digest differs from verified report")
        class HTTP(ThreadingHTTPServer):
            daemon_threads = True
            timeout = 2
            def process_request(self, request_socket, address):
                if not semaphore.acquire(blocking=False):
                    request_socket.close()
                    return
                try:
                    super().process_request(request_socket, address)
                except Exception:
                    semaphore.release()
                    raise
            def process_request_thread(self, request_socket, address):
                try:
                    super().process_request_thread(request_socket, address)
                finally:
                    semaphore.release()
        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(10)
            def log_message(self, *args):
                pass
            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length", "-1"))
                    if self.path != "/runnable" or not 0 < length <= 1 << 20 or "Transfer-Encoding" in self.headers:
                        raise ValueError("bounded POST required")
                    value = strict_json(self.rfile.read(length))
                    sent = public_request(value, identity["keccak"])
                    bits = response(post(sent), sent)
                    reply = {"results": [{"score": b} for b in bits], "foundation_hash": FOUNDATION,
                             "adapter_hash": identity["keccak"], "tokenizer_hash": TOKENIZER}
                    if "input_root" in value:
                        reply["input_root"] = sent["input_root"]
                    status = 200
                except (ValueError, TypeError, KeyError, OSError):
                    reply, status = {"error": "request refused"}, 400
                raw = json.dumps(reply, separators=(",", ":"), allow_nan=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(raw)
        http = HTTP(("127.0.0.1", plan["local_port"]), Handler)
        # Preflight the actual engine against the audit job's exact public vectors.
        ready_report = json.loads((root / "registration-ready.json").read_bytes())
        for vector in ready_report["runtime"]["vectors"]:
            if score([vector["tokens"]]) != [vector["score_bits"]]:
                raise ValueError("serving reference bits differ from independently verified package")
        dump(root / "backend-ready.json", {"pid": os.getpid(), "local_port": plan["local_port"], "package": identity,
             "reference_vectors_passed": len(ready_report["runtime"]["vectors"])})
        deadline = time.monotonic() + 5 * 3600
        try:
            while time.monotonic() < deadline:
                http.handle_request()
        finally:
            http.server_close()


def prepare(root, package_url, package_sha, package_keccak, report_url, report_sha, slot, port, runnable):
    remote_only()
    validate_reservation(slot, port, runnable)
    root.mkdir(parents=True, exist_ok=True)
    immutable_url(package_url)
    immutable_url(report_url)
    fetch(report_url, root / "registration-ready.json", report_sha, 256 << 10)
    report = strict_json((root / "registration-ready.json").read_bytes())
    if report.get("status") != "REGISTRATION_READY":
        raise ValueError("passing real audit report required")
    ready(report)
    if report["package_sha256"] != package_sha or report["package_keccak"] != package_keccak or report["serving_commit"] != os.environ["GITHUB_SHA"]:
        raise ValueError("verified package/report/serving-commit identity mismatch")
    fetch(package_url, root / "ACADEMIC_SEARCH.safetensors", package_sha, 64 << 20)
    for name, expected in MODEL_FILES.items():
        fetch(RELEASE + name, root / "foundation" / name, expected)
    kit = prepare_kit(root)
    compile_tools(root, kit)
    contract = lease_module()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        local_port = probe.getsockname()[1]
    dump(root / "serve-plan.json", {"slot": slot, "forward_port": port, "runnable_url": runnable, "local_port": local_port,
         "package_sha256": package_sha, "package_keccak": package_keccak, "report_sha256": report_sha})
    with (root / "backend.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(HERE / "serving.py"), "backend", "--root", str(root)], stdout=log, stderr=log)
    deadline = time.monotonic() + 300
    while not (root / "backend-ready.json").is_file():
        if process.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError("actual serving preflight failed or timed out")
        time.sleep(1)
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "academic-forward", "-f", str(root / "forward-key")], check=True, timeout=30)
    os.chmod(root / "forward-key", 0o600)
    row = {"family": "ACADEMIC_SEARCH", "owner": contract.DEFAULT_OWNER, "slot": slot, "forward_port": port,
           "runnable_url": runnable, "keccak": package_keccak, "sha256": package_sha, "existing_registration_id": 0,
           **{key + "_proof": {"sha256": report_sha} for key in ("candidate", "runtime", "source")}}
    # Use the existing common.lease_for bytes, not another signer or lease format.
    lease = contract.lease_for(row, (root / "forward-key.pub").read_text().strip(), "academic-" + os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"])
    contract.validate_lease(lease, row)
    dump(root / "restricted-public-lease.json", lease)
    (root / "known_hosts").write_text(contract.HOST_KEY)


def forward(root):
    remote_only()
    contract = lease_module()
    plan = json.loads((root / "serve-plan.json").read_bytes())
    deadline = time.monotonic() + 5 * 3600
    lease = json.loads((root / "restricted-public-lease.json").read_bytes())
    deadline = time.monotonic() + max(0, contract.utc(lease["expires_at"]).timestamp() - time.time())
    arguments = ["ssh", "-F", "/dev/null", "-N", "-T", "-i", str(root / "forward-key"), "-o", "IdentitiesOnly=yes",
                 "-o", "IdentityAgent=none", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes",
                 "-o", "UserKnownHostsFile=" + str(root / "known_hosts"), "-o", "GlobalKnownHostsFile=/dev/null", "-o", "ExitOnForwardFailure=yes",
                 "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3", "-o", "ForwardAgent=no", "-o", "ForwardX11=no",
                 "-o", "PermitLocalCommand=no", "-o", "RequestTTY=no", "-o", "LogLevel=ERROR",
                 "-R", "127.0.0.1:" + str(plan["forward_port"]) + ":127.0.0.1:" + str(plan["local_port"]), "telegraph-verify@" + contract.HOST]
    # Transport retries are idempotent. A live forwarding process is never replaced.
    while time.monotonic() < deadline:
        process = subprocess.Popen(arguments, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(2)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
        time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "backend", "forward"))
    parser.add_argument("--root", type=Path, required=True)
    for field in ("package-url", "package-sha256", "package-keccak", "report-url", "report-sha256", "runnable-url"):
        parser.add_argument("--" + field)
    parser.add_argument("--slot", type=int)
    parser.add_argument("--forward-port", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root, args.package_url, args.package_sha256, args.package_keccak, args.report_url, args.report_sha256, args.slot, args.forward_port, args.runnable_url)
    elif args.mode == "backend":
        backend(args.root)
    else:
        forward(args.root)
