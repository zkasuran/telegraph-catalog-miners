# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Exact remote executor and public restricted forwarding lease. Never run locally."""
import argparse
import http.client
import json
import os
import re
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from contracts import response_bits
from data import FAMILY, sha
from policy import remote_only
from remote_prepare import prepare, fetch
from registration_common import lease_for, validate_lease, DEFAULT_OWNER, HOST, HOST_KEY, strict_json


def public_url(value):
    prefix = "https://raw.githubusercontent.com/zkasuran/telegraph-catalog-miners/"
    if not value.startswith(prefix) or not re.fullmatch(r"[a-f0-9]{40}/verification/owners/smart_contract_audit/[A-Za-z0-9_.-]+", value[len(prefix):]):
        raise ValueError("owned immutable public URL required")
    return value


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("prepare", "run"))
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--package-url")
    p.add_argument("--report-url")
    p.add_argument("--report-sha256")
    p.add_argument("--package-sha256")
    p.add_argument("--slot", type=int)
    p.add_argument("--forward-port", type=int)
    p.add_argument("--runnable-url")
    return p.parse_args()


def prepare_serving(args):
    root = args.root
    root.mkdir(exist_ok=True)
    if not 1 <= args.slot <= 6 or args.forward_port != 48600 + args.slot:
        raise ValueError("free existing proxy slot mismatch")
    if not re.fullmatch(r"https://tga1\.zkasuran\.dev/_[A-Za-z0-9_-]{1,64}/runnable", args.runnable_url):
        raise ValueError("existing proxy runnable URL required")
    for digest in (args.report_sha256, args.package_sha256):
        if not re.fullmatch(r"[a-f0-9]{64}", digest or ""):
            raise ValueError("verified SHA-256 required")
    report_path, package = root / "registration-ready.json", root / "candidate.safetensors"
    fetch(public_url(args.report_url), report_path, args.report_sha256, 1 << 20)
    fetch(public_url(args.package_url), package, args.package_sha256, 64 << 20)
    report = json.loads(report_path.read_text())
    if report["family"] != FAMILY or report["status"] != "REGISTRATION_READY" or report["package_sha256"] != args.package_sha256 or report["serving_commit"] != os.environ["GITHUB_SHA"]:
        raise ValueError("verified report/package/serving source identity mismatch")
    foundation, _, kit, hashes, _ = prepare(root, training=False)
    for field, filename in (("foundation_sha256", "foundation_lora_fp32.onnx"), ("engine_sha256", "foundation_engine.wasm"), ("tokenizer_sha256", "tokenizer.json")):
        if report[field] != hashes[filename]:
            raise ValueError("foundation runtime report mismatch")
    subprocess.run(["go", "build", "-trimpath", "-o", str(root / "adapterexec"), "./cmd/adapterexec"], cwd=kit, check=True, timeout=600)
    key = root / "forward-key"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key), "-C", "smart-audit-forward"], check=True, timeout=15)
    key.chmod(0o600)
    (root / "known_hosts").write_text(HOST_KEY)
    row = {"family": FAMILY, "owner": DEFAULT_OWNER, "slot": args.slot, "forward_port": args.forward_port,
           "runnable_url": args.runnable_url, "keccak": report["package_keccak"], "sha256": args.package_sha256,
           "existing_registration_id": 0, **{kind + "_proof": {"sha256": args.report_sha256} for kind in ("candidate", "runtime", "source")}}
    lease = lease_for(row, key.with_suffix(".pub").read_text(), "smart-audit-" + os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"])
    validate_lease(lease, row)
    (root / "lease.json").write_text(json.dumps(lease, indent=2) + "\n")
    (root / "state.json").write_text(json.dumps({"row": row, "foundation": hashes, "report_sha256": args.report_sha256}) + "\n")
    print("Pinned exact executor prepared. Only the restricted public lease may be uploaded.")


def serve(root):
    state = json.loads((root / "state.json").read_text())
    row, hashes = state["row"], state["foundation"]
    lease = json.loads((root / "lease.json").read_text())
    validate_lease(lease, row)
    report_raw = (root / "registration-ready.json").read_bytes()
    if sha(report_raw) != state["report_sha256"] or sha((root / "candidate.safetensors").read_bytes()) != row["sha256"]:
        raise ValueError("verified serving bytes changed")
    report = json.loads(report_raw)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        backend_port = probe.getsockname()[1]
    env = {**os.environ, "FOUNDATION_MODEL_PATH": str(root / "foundation/foundation_lora_fp32.onnx"),
           "FOUNDATION_MODEL_HASH": hashes["foundation_lora_fp32.onnx"],
           "FOUNDATION_ENGINE_WASM_PATH": str(root / "foundation/foundation_engine.wasm"),
           "FOUNDATION_ENGINE_WASM_HASH": hashes["foundation_engine.wasm"]}
    backend_log = (root / "executor.log").open("w")
    process = subprocess.Popen([str(root / "adapterexec"), "--package", str(root / "candidate.safetensors"), "--listen", f"127.0.0.1:{backend_port}"], env=env, stdout=backend_log, stderr=backend_log)
    forward = None
    expected = {"foundation_hash": hashes["foundation_lora_fp32.onnx"], "tokenizer_hash": hashes["tokenizer.json"], "adapter_hash": row["keccak"]}

    def score(body):
        connection = http.client.HTTPConnection("127.0.0.1", backend_port, timeout=600)
        try:
            connection.request("POST", "/runnable", json.dumps(body), {"Content-Type": "application/json"})
            response = connection.getresponse()
            raw = response.read((1 << 20) + 1)
            if response.status != 200 or len(raw) > 1 << 20:
                raise ValueError("backend response bound")
            parsed = json.loads(raw)
            response_bits(parsed, body)
            return parsed
        finally:
            connection.close()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *args):
            return

        def do_POST(self):
            try:
                if self.path != "/runnable" or self.headers.get("Transfer-Encoding"):
                    raise ValueError("path or framing refused")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1 << 20:
                    raise ValueError("body size refused")
                body = strict_json(self.rfile.read(size))
                if set(body) != {"batch", "input_root", *expected} or any(body[k] != v for k, v in expected.items()):
                    raise ValueError("literal identity refused")
                if not isinstance(body["batch"], list) or not 1 <= len(body["batch"]) <= 100:
                    raise ValueError("batch refused")
                for item in body["batch"]:
                    if set(item) != {"tokens"} or not isinstance(item["tokens"], list) or not 1 <= len(item["tokens"]) <= 512 or any(type(t) is not int or not 0 <= t < 50368 for t in item["tokens"]):
                        raise ValueError("tokens refused")
                payload, status = score(body), 200
            except (ValueError, TypeError, KeyError, OSError):
                payload, status = {"error": "request refused"}, 400
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise ValueError("exact executor failed to load")
            try:
                with socket.create_connection(("127.0.0.1", backend_port), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise ValueError("executor start timeout")
        # The kit computes the input root. Do not normalize a substitute identity.
        wire_dir = root / ("telegraph-wasm-baseline-" + json.loads((Path(__file__).parent / "sources.json").read_text())["kit"]["revision"]) / "cmd/owner-wire"
        wire_dir.mkdir(exist_ok=True)
        (wire_dir / "main.go").write_bytes((Path(__file__).parent / "wire_root.go").read_bytes())
        subprocess.run(["go", "build", "-trimpath", "-o", str(root / "wire-root"), "./cmd/owner-wire"], cwd=wire_dir.parent.parent, check=True, timeout=600)
        vectors = report["runtime"]["vectors"]
        batch = [v["tokens"] for v in vectors]
        computed = subprocess.run([str(root / "wire-root"), str(root / "candidate.safetensors")], input=json.dumps(batch), text=True, capture_output=True, check=True, timeout=30)
        request = {"batch": [{"tokens": ids} for ids in batch], **expected, **json.loads(computed.stdout)}
        result = score(request)
        if [r["score"] for r in result["results"]] != [v["score_bits"] for v in vectors]:
            raise ValueError("serving executor differs from independent audit vectors")
        class BoundedServer(ThreadingHTTPServer):
            daemon_threads = True
            slots = threading.BoundedSemaphore(8)

            def process_request(self, request, client_address):
                if not self.slots.acquire(blocking=False):
                    self.shutdown_request(request)
                    return
                try:
                    super().process_request(request, client_address)
                except BaseException:
                    self.slots.release()
                    raise

            def process_request_thread(self, request, client_address):
                try:
                    super().process_request_thread(request, client_address)
                finally:
                    self.slots.release()

        proxy = BoundedServer(("127.0.0.1", 0), Handler)
        proxy.timeout = 10
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        port = proxy.server_address[1]
        from datetime import datetime
        remaining = datetime.fromisoformat(lease["expires_at"]).timestamp() - time.time()
        stop = time.monotonic() + max(0, remaining)
        while time.monotonic() < stop:
            if process.poll() is not None:
                raise ValueError("serving executor exited")
            if forward is None or forward.poll() is not None:
                forward = subprocess.Popen(["ssh", "-F", "/dev/null", "-N", "-T", "-i", str(root / "forward-key"),
                    "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                    "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + str(root / "known_hosts"),
                    "-o", "GlobalKnownHostsFile=/dev/null", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=20",
                    "-o", "ServerAliveCountMax=3", "-o", "ForwardAgent=no", "-o", "ForwardX11=no", "-o", "LogLevel=ERROR",
                    "-R", f"127.0.0.1:{row['forward_port']}:127.0.0.1:{port}", "telegraph-verify@" + HOST], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(10)
    finally:
        for child in (forward, process):
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
        backend_log.close()


def main():
    remote_only()
    args = arguments()
    prepare_serving(args) if args.mode == "prepare" else serve(args.root)


if __name__ == "__main__":
    main()
