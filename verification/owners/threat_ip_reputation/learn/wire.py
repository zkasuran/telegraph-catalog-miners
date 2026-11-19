# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Strict bounded wire front end, without changing reference-engine math."""
import http.client
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY = 1 << 20
MAX_BATCH = 100
MAX_TOKENS = 512
VOCAB = 50368


def input_root(batch):
    leaves = [str(index) + ":" + hashlib.sha256(json.dumps({"input_ids": item["tokens"], "attention_mask": [1] * len(item["tokens"])}, separators=(",", ":")).encode()).hexdigest()
              for index, item in enumerate(batch)]
    return hashlib.sha256("\n".join(sorted(leaves)).encode()).hexdigest() if leaves else ""


def unique(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError("duplicate JSON field")
        value[key] = item
    return value


def decode(raw):
    if len(raw) > MAX_BODY:
        raise ValueError("wire body ceiling")
    try:
        return json.loads(raw, object_pairs_hook=unique, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except RecursionError as exc:
        raise ValueError("wire nesting ceiling") from exc


def validate_request(value, expected):
    if not isinstance(value, dict) or set(value) - {"batch", "input_root", *expected} or not {"batch", *expected} <= set(value):
        raise ValueError("literal token-only request fields required")
    for key, identity in expected.items():
        if not isinstance(value[key], str) or not re.fullmatch(r"[a-f0-9]{64}", value[key]) or value[key] != identity:
            raise ValueError("literal pinned identity mismatch")
    batch = value["batch"]
    if not isinstance(batch, list) or len(batch) > MAX_BATCH:
        raise ValueError("batch count ceiling")
    for item in batch:
        if not isinstance(item, dict) or set(item) != {"tokens"} or not isinstance(item["tokens"], list) or not 1 <= len(item["tokens"]) <= MAX_TOKENS:
            raise ValueError("token array contract")
        if any(type(token) is not int or not 0 <= token < VOCAB for token in item["tokens"]):
            raise ValueError("token ID refusal")
    if "input_root" in value and value["input_root"] != input_root(batch):
        raise ValueError("literal input root mismatch")
    return value


def validate_response(value, request, expected):
    if not isinstance(value, dict) or set(value) - {"results", "input_root", *expected} or not {"results", *expected} <= set(value) or any(value.get(key) != identity for key, identity in expected.items()):
        raise ValueError("reference response literal identity/field mismatch")
    if request.get("input_root") and value.get("input_root") != request["input_root"]:
        raise ValueError("reference input root mismatch")
    results = value["results"]
    if not isinstance(results, list) or len(results) != len(request["batch"]):
        raise ValueError("reference result count mismatch")
    for item in results:
        if not isinstance(item, dict) or set(item) != {"score"}:
            raise ValueError("reference score field mismatch")
        bits = item["score"]
        if type(bits) is not int or not 0 <= bits < 1 << 32 or bits & 0x7f800000 == 0x7f800000:
            raise ValueError("reference finite score bits required")
    if "input_root" in request and not request["input_root"]:
        value = {**value, "input_root": ""}
    return value


class Backend:
    def __init__(self, port):
        self.port = port

    def score(self, value):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
        try:
            connection.request("POST", "/runnable", json.dumps(value).encode(), {"Content-Type": "application/json", "Connection": "close"})
            response = connection.getresponse()
            raw = response.read(MAX_BODY + 1)
            if response.status != 200:
                raise ValueError("reference backend failed")
            return decode(raw)
        finally:
            connection.close()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    request_queue_size = 8

    def __init__(self, address, expected, backend):
        self.expected, self.backend = expected, backend
        self.lock = threading.BoundedSemaphore(1)
        super().__init__(address, Handler)

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(10)
        return connection, address


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def respond(self, code, value):
        raw = json.dumps(value).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True

    def do_POST(self):
        if self.path != "/runnable" or self.headers.get("Transfer-Encoding") or len(self.headers.get_all("Content-Length", [])) != 1:
            self.respond(400, {"error": "invalid_request"})
            return
        try:
            size = int(self.headers["Content-Length"])
            if not 0 < size <= MAX_BODY:
                raise ValueError("wire request ceiling")
            request = validate_request(decode(self.rfile.read(size)), self.server.expected)
        except (ValueError, OSError):
            self.respond(400, {"error": "invalid_request"})
            return
        if not self.server.lock.acquire(blocking=False):
            self.respond(503, {"error": "busy"})
            return
        try:
            result = validate_response(self.server.backend.score(request), request, self.server.expected)
            self.respond(200, result)
        except (ValueError, OSError):
            self.respond(502, {"error": "reference_failure"})
        finally:
            self.server.lock.release()
