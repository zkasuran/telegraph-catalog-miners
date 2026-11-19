# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded literal-wire frontend to the unmodified reference engine. Remote only."""
import http.client
import json
import os
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from contracts import FOUNDATION, check_wire, remote_only, strict_json

MAX_BODY = 1 << 20
MAX_HEADERS = 16 << 10
MAX_BATCH = 100
MAX_TOKENS = 512


def validate_request(value, package_keccak):
    identity = {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"], "adapter_hash": package_keccak,
                "tokenizer_hash": FOUNDATION["tokenizer.json"]}
    if not isinstance(value, dict) or set(value) - {"batch", "input_root", *identity} or not {"batch", *identity} <= set(value):
        raise ValueError("literal runnable envelope mismatch")
    if any(value[key] != wanted for key, wanted in identity.items()):
        raise ValueError("literal runnable identity mismatch")
    batch = value["batch"]
    if not isinstance(batch, list) or not 1 <= len(batch) <= MAX_BATCH:
        raise ValueError("runnable batch bound")
    for item in batch:
        if not isinstance(item, dict) or set(item) != {"tokens"}:
            raise ValueError("runnable accepts literal token arrays only")
        tokens = item["tokens"]
        if (not isinstance(tokens, list) or not 1 <= len(tokens) <= MAX_TOKENS
                or any(type(token) is not int or not 0 <= token < 50368 for token in tokens)):
            raise ValueError("runnable token bound")
    if "input_root" in value:
        root = value["input_root"]
        if not isinstance(root, str) or len(root) > 66:
            raise ValueError("input-root wire bound")
    return value


class HeaderBudget:
    def __init__(self, stream):
        self.stream, self.remaining = stream, MAX_HEADERS

    def readline(self, limit=-1):
        limit = self.remaining + 1 if limit < 0 else min(limit, self.remaining + 1)
        raw = self.stream.readline(limit)
        self.remaining -= len(raw)
        if self.remaining < 0:
            raise http.client.LineTooLong("request header ceiling")
        return raw


class ReferenceExecutor:
    def __init__(self, root, package, package_keccak):
        remote_only()
        self.root, self.package_keccak = root, package_keccak
        self.lock = threading.Lock()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        environment = {"PATH": os.environ["PATH"], "LANG": "C.UTF-8",
            "FOUNDATION_MODEL_PATH": str(root/"foundation/foundation_lora_fp32.onnx"),
            "FOUNDATION_MODEL_HASH": FOUNDATION["foundation_lora_fp32.onnx"],
            "FOUNDATION_ENGINE_WASM_PATH": str(root/"foundation/foundation_engine.wasm"),
            "FOUNDATION_ENGINE_WASM_HASH": FOUNDATION["foundation_engine.wasm"]}
        self.log = (root/(package.stem+"-engine.log")).open("w")
        self.process = subprocess.Popen([str(root/"adapterexec"), "--package", str(package), "--listen", f"127.0.0.1:{self.port}"],
                                        env=environment, stdout=self.log, stderr=self.log)
        deadline = time.monotonic()+180
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.close()
                raise RuntimeError("actual reference engine did not load")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=1):
                    return
            except OSError:
                time.sleep(1)
        self.close()
        raise RuntimeError("reference engine readiness timeout")

    def score(self, request):
        validate_request(request, self.package_keccak)
        if self.process.poll() is not None:
            raise RuntimeError("reference engine is no longer running")
        if not self.lock.acquire(timeout=10):
            raise RuntimeError("reference scorer concurrency bound")
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
        try:
            connection.request("POST", "/runnable", json.dumps(request, separators=(",", ":")), {"Content-Type": "application/json", "Connection": "close"})
            response = connection.getresponse()
            raw = response.read(MAX_BODY+1)
            if response.status != 200 or len(raw) > MAX_BODY:
                raise ValueError("reference engine response ceiling/status")
            value = strict_json(raw)
            check_wire(value, len(request["batch"]), self.package_keccak)
            if "input_root" in request and value.get("input_root") != request["input_root"]:
                raise ValueError("literal engine batch root mismatch")
            return value
        finally:
            connection.close()
            self.lock.release()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.log.close()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    request_queue_size = 4

    def __init__(self, address, backend):
        self.backend = backend
        self.connections = threading.BoundedSemaphore(4)
        super().__init__(address, Handler)

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)
        self.input_timer = threading.Timer(20, self.expire_input)
        self.input_timer.daemon = True
        self.input_timer.start()

    def expire_input(self):
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def finish(self):
        self.input_timer.cancel()
        super().finish()

    def parse_request(self):
        original = self.rfile
        self.rfile = HeaderBudget(original)
        try:
            return super().parse_request()
        finally:
            self.rfile = original

    def reply(self, code, value):
        raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True

    def do_POST(self):
        try:
            if self.path != "/runnable" or self.headers.get("Transfer-Encoding"):
                raise ValueError("unsupported request path or encoding")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not lengths[0].isdigit() or not 1 <= int(lengths[0]) <= MAX_BODY:
                raise ValueError("request body byte bound")
            raw = self.rfile.read(int(lengths[0]))
            if len(raw) != int(lengths[0]):
                raise ValueError("incomplete request body")
            request = validate_request(strict_json(raw), self.server.backend.package_keccak)
            self.input_timer.cancel()
            self.reply(200, self.server.backend.score(request))
        except (ValueError, UnicodeError, RecursionError):
            self.reply(400, {"error": "request refused"})
        except (OSError, RuntimeError):
            self.reply(503, {"error": "scoring unavailable"})

    def log_message(self, *_):
        pass
