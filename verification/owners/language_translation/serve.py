# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only restricted existing-proxy lease for the exact verified package."""
import argparse
import ast
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from contracts import (
    FAMILY,
    PACKAGE_KECCAK,
    PACKAGE_SHA,
    SOURCE_SHA,
    read_asset,
    remote_only,
    sha,
    verify_assets,
)
from replay import fetch, prepare, write
from wire import request_ok, response_ok


def serve(args):
    remote_only()
    verify_assets()
    root = args.root.resolve()
    prepare(root)
    report_path = root / "registration-ready.json"
    if args.report_path:
        if sha(args.report_path) != args.report_sha:
            raise ValueError("downloaded real artifact report hash mismatch")
        import shutil
        shutil.copyfile(args.report_path, report_path)
    else:
        fetch(args.report_url, report_path, args.report_sha, 256 << 10)
    report = json.loads(report_path.read_text())
    if (report["status"] != "REGISTRATION_READY" or report["family"] != FAMILY or report["package_sha256"] != PACKAGE_SHA
            or report["package_keccak"] != PACKAGE_KECCAK or report["serving_commit"] != os.environ["GITHUB_SHA"]
            or report["source"]["source_manifest_sha256"] != SOURCE_SHA or not report["runtime"]["reference_engine_passed"]):
        raise ValueError("serving report/commit/package mismatch")
    if not 1 <= args.slot <= 100 or not 1024 <= args.forward_port <= 65535 or not args.runnable_url.startswith("https://tga1.zkasuran.dev/_"):
        raise ValueError("reserved endpoint identity/port refused")
    raw = read_asset("reference-common.py.txt")
    tree = ast.parse(raw.decode())
    names = {"now", "public_key", "lease_for"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    import base64
    import re
    import struct
    scope = {"base64": base64, "re": re, "struct": struct, "datetime": datetime, "timezone": timezone,
             "COMMIT": re.compile(r"[0-9a-f]{40}"), "WINDOW": 5 * 3600}
    # Execute only the unchanged, hash-verified public lease contract functions.
    exec(compile(ast.Module(body=selected, type_ignores=[]), "pinned-public-lease", "exec"), scope)  # noqa: S102
    key = root / "restricted-forward-key"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key), "-C", "translation-runtime-lease"], check=True, timeout=20)
    known = root / "known_hosts"
    known.write_text("98.95.110.88 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMRYtmP2ESjH1ah2MI3eqUmEgtVYeyR81RAbr0MB/r0G\n")
    proof = {"url": args.report_url or "https://github.com/zkasuran/telegraph-catalog-miners/actions/runs/" + str(report["run_id"]), "sha256": args.report_sha}
    row = {"family": FAMILY, "owner": "0x8b224783fe5b3c52b7db0cb9b1754f8812b75287", "slot": args.slot,
           "forward_port": args.forward_port, "runnable_url": args.runnable_url, "sha256": PACKAGE_SHA, "keccak": PACKAGE_KECCAK,
           "existing_registration_id": 0, "candidate_proof": proof, "runtime_proof": proof, "source_proof": proof}
    lease = scope["lease_for"](row, key.with_suffix(".pub").read_text(), "translation-lease", os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"])
    env = dict(os.environ)
    from contracts import FOUNDATION
    env.update({"FOUNDATION_MODEL_PATH": str(root / "foundation/foundation_lora_fp32.onnx"), "FOUNDATION_MODEL_HASH": FOUNDATION["foundation_lora_fp32.onnx"],
                "FOUNDATION_ENGINE_WASM_PATH": str(root / "foundation/foundation_engine.wasm"), "FOUNDATION_ENGINE_WASM_HASH": FOUNDATION["foundation_engine.wasm"]})
    port = 38693
    log = (root / "serving.log").open("w")
    engine = subprocess.Popen([str(root / "adapterexec"), "--package", str(root / (FAMILY + ".safetensors")), "--listen", f"127.0.0.1:{port}"], env=env, stdout=log, stderr=log)
    try:
        import socket
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if engine.poll() is not None:
                raise RuntimeError("identical reference executor failed readiness")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError("serving readiness ceiling")
        import threading
        import urllib.request
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        slots = threading.BoundedSemaphore(8)
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if not slots.acquire(blocking=False):
                    self.send_error(503)
                    return
                try:
                    self.connection.settimeout(10)
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 1 << 20:
                        raise ValueError("request byte ceiling")
                    raw_request = self.rfile.read(length)
                    request = request_ok(json.loads(raw_request))
                    operation = urllib.request.Request(f"http://127.0.0.1:{port}/runnable", data=raw_request, headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(operation, timeout=600) as reply:
                        raw_response = reply.read((1 << 20) + 1)
                    if len(raw_response) > 1 << 20:
                        raise ValueError("response byte ceiling")
                    response_ok(json.loads(raw_response), len(request["batch"]))
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(raw_response)
                except (ValueError, OSError, TypeError):
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error":"bounded exact runtime request refused"}')
                finally:
                    slots.release()
            def log_message(self, *values):
                pass
        public_port = port + 1
        proxy = ThreadingHTTPServer(("127.0.0.1", public_port), Handler)
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        # Only this restricted public lease is uploaded. The private key remains
        # ephemeral on the hosted runner and is never put in an artifact.
        write(root / "public-lease.json", lease)
        forward = None
        end = time.monotonic() + 4.5 * 3600
        while time.monotonic() < end:
            if engine.poll() is not None:
                raise RuntimeError("reference serving process ended")
            if forward is None or forward.poll() is not None:
                forward = subprocess.Popen(["ssh", "-F", "/dev/null", "-i", str(key), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                    "-o", "UserKnownHostsFile=" + str(known), "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3",
                    "-N", "-R", f"127.0.0.1:{args.forward_port}:127.0.0.1:{public_port}", "telegraph-verify@98.95.110.88"], stdout=log, stderr=log)
            time.sleep(5)
    finally:
        if "forward" in locals() and forward is not None and forward.poll() is None:
            forward.terminate()
            forward.wait(timeout=20)
        if "proxy" in locals():
            proxy.shutdown()
            proxy.server_close()
        engine.terminate()
        engine.wait(timeout=20)
        log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--report-url", default="")
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--report-sha", required=True)
    parser.add_argument("--slot", required=True, type=int)
    parser.add_argument("--forward-port", required=True, type=int)
    parser.add_argument("--runnable-url", required=True)
    serve(parser.parse_args())
