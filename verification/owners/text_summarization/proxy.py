# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""A bounded token-only front end for the real reference executor."""
import http.client
import math
import re
import socket
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from common import (
    FOUNDATION,
    MAX_BATCH,
    MAX_BODY,
    MAX_CONNECTIONS,
    MAX_HEADERS,
    MAX_TOKENS,
    MAX_TOTAL_TOKENS,
    READ_TIMEOUT,
    SCORE_TIMEOUT,
    VOCAB_SIZE,
    json_bytes,
    sha,
    strict_json,
)


def identity(row):
    return {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
            "adapter_hash": row["keccak"], "tokenizer_hash": FOUNDATION["tokenizer.json"]}


def input_root(batch):
    # [INPUT] remote.BatchInputRoot: sorted "index:InputHash" leaves joined by LF.
    leaves = [f"{i}:" + sha(json_bytes({"input_ids": item["tokens"],
                                     "attention_mask": [1] * len(item["tokens"])}))
              for i, item in enumerate(batch)]
    return sha("\n".join(sorted(leaves)).encode()) if leaves else ""


def normalize_hash(value):
    if not isinstance(value, str):
        # Wrong types and malformed hashes must follow the same HTTP rejection path.
        raise ValueError("invalid score identity")  # noqa: TRY004
    normalized = value.lower().removeprefix("0x")
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError("invalid score identity")
    return normalized


def validate_request(value, row):
    expected = identity(row)
    allowed = {"batch", "input_root", *expected}
    if not isinstance(value, dict) or set(value) - allowed or not {"batch", *expected} <= set(value):
        raise ValueError("invalid runnable request fields")
    for name, wanted in expected.items():
        if value[name] != wanted:
            raise ValueError("runnable request identity mismatch")
    batch = value["batch"]
    if not isinstance(batch, list) or len(batch) > MAX_BATCH:
        raise ValueError("invalid runnable batch count")
    total = 0
    for item in batch:
        if not isinstance(item, dict) or set(item) != {"tokens"}:
            raise ValueError("runnable accepts token arrays only")
        tokens = item["tokens"]
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= MAX_TOKENS:
            raise ValueError("invalid runnable token count")
        total += len(tokens)
        if total > MAX_TOTAL_TOKENS or any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in tokens):
            raise ValueError("invalid runnable token id")
    root = input_root(batch)
    if "input_root" in value:
        received = value["input_root"]
        if received != root:
            raise ValueError("runnable input root mismatch")
    return {"batch": batch, **expected, **({"input_root": root} if root else {})}


def score_bits(value):
    if type(value) is not int or not 0 <= value < 1 << 32:
        raise ValueError("missing or invalid FP32 score")
    decoded = struct.unpack("<f", struct.pack("<I", value))[0]
    if not math.isfinite(decoded):
        raise ValueError("nonfinite FP32 score")
    return value


def validate_response(value, request, row, errors=False):
    expected = identity(row)
    if not isinstance(value, dict) or set(value) - {"results", "input_root", *expected}:
        raise ValueError("invalid runtime response fields")
    for name, wanted in expected.items():
        if value.get(name) != wanted:
            raise ValueError("runtime response identity mismatch")
    if request.get("input_root") and value.get("input_root") != request["input_root"]:
        raise ValueError("runtime response input root mismatch")
    results = value.get("results")
    if not isinstance(results, list) or len(results) != len(request["batch"]):
        raise ValueError("missing, duplicate or extra runtime results")
    safe = []
    for item in results:
        if not isinstance(item, dict):
            # All malformed executor results share the ValueError/fatal contract.
            raise ValueError("invalid runtime result")  # noqa: TRY004
        if set(item) == {"score"}:
            safe.append({"score": score_bits(item["score"])})
        elif errors and set(item) == {"error"} and isinstance(item["error"], str) and 0 < len(item["error"]) <= 4096:
            # Used only for fixed source-only negative startup probes, never exposed.
            safe.append({"error": "refused"})
        else:
            raise ValueError("missing or conflicting runtime score")
    return {"results": safe, **expected, **({"input_root": value["input_root"]} if request.get("input_root") else {})}


class Backend:
    def __init__(self, port, timeout=SCORE_TIMEOUT):
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("invalid loopback executor port")
        self.port = port
        self.timeout = timeout

    def score(self, request):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)

        def abort():
            if connection.sock is not None:
                try:
                    connection.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            connection.close()

        timer = threading.Timer(self.timeout, abort)
        timer.daemon = True
        timer.start()
        try:
            connection.request("POST", "/runnable", json_bytes(request),
                               {"Content-Type": "application/json", "Connection": "close"})
            response = connection.getresponse()
            raw = response.read(MAX_BODY + 1)
            if response.status != 200 or len(raw) > MAX_BODY:
                raise ValueError("runtime response status or byte ceiling exceeded")
            return strict_json(raw)
        finally:
            timer.cancel()
            connection.close()


class HeaderBudget:
    def __init__(self, stream):
        self.stream = stream
        self.remaining = MAX_HEADERS

    def readline(self, limit=-1):
        size = self.remaining + 1 if limit < 0 else min(limit, self.remaining + 1)
        line = self.stream.readline(size)
        self.remaining -= len(line)
        if self.remaining < 0:
            raise http.client.LineTooLong("bounded request headers")
        return line


class BoundedServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    request_queue_size = MAX_CONNECTIONS

    def __init__(self, address, row, backend):
        self.row = row
        self.backend = backend
        self.connections = threading.BoundedSemaphore(MAX_CONNECTIONS)
        self.scoring = threading.BoundedSemaphore(1)
        self.fatal = threading.Event()
        super().__init__(address, Handler)

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(READ_TIMEOUT)
        return connection, address

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            try:
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            except OSError:
                pass
            self.shutdown_request(request)
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

    def handle_error(self, request, client_address):
        # No request, address, token list or stack trace goes to logs.
        pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "CandidateEndpoint"
    sys_version = ""

    def setup(self):
        super().setup()

        def abort_read():
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        self.read_timer = threading.Timer(READ_TIMEOUT, abort_read)
        self.read_timer.daemon = True
        self.read_timer.start()

    def finish(self):
        self.read_timer.cancel()
        super().finish()

    def log_message(self, format, *args):
        pass

    def parse_request(self):
        if len(self.raw_requestline) > 8192:
            self.requestline, self.request_version, self.command = "", "HTTP/1.1", ""
            self.send_error(414)
            return False
        original = self.rfile
        self.rfile = HeaderBudget(original)
        try:
            return super().parse_request()
        finally:
            self.rfile = original

    def handle_expect_100(self):
        self.send_error(417)
        return False

    def send_error(self, code, message=None, explain=None):
        self.send_json(code, {"error": "request refused"})

    def send_json(self, status, value):
        self.read_timer.cancel()
        raw = json_bytes(value)
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.connection.settimeout(READ_TIMEOUT)
        self.wfile.write(raw)

    def do_POST(self):
        if self.path not in ("/runnable", urlsplit(self.server.row["runnable_url"]).path):
            self.send_error(404)
            return
        if (self.headers.get_all("Transfer-Encoding") or self.headers.get_all("Content-Encoding") or
                len(self.headers.get_all("Content-Length", [])) != 1 or
                len(self.headers.get_all("Content-Type", [])) != 1 or
                self.headers.get("Content-Type", "").lower() != "application/json"):
            self.send_error(400)
            return
        length = self.headers["Content-Length"]
        if not re.fullmatch(r"[0-9]{1,8}", length) or not 1 <= int(length) <= MAX_BODY:
            self.send_error(413)
            return
        try:
            body = self.rfile.read(int(length))
            if len(body) != int(length):
                raise ValueError("incomplete runnable body")
            value = strict_json(body)
            request = validate_request(value, self.server.row)
        except (ValueError, TypeError, KeyError, RecursionError, OSError):
            self.send_error(400)
            return
        self.read_timer.cancel()
        if self.server.fatal.is_set() or not self.server.scoring.acquire(blocking=False):
            self.send_error(503)
            return
        try:
            response = validate_response(self.server.backend.score(request), request, self.server.row)
            if "input_root" not in value:
                response.pop("input_root", None)
        except (ValueError, TypeError, KeyError, RecursionError, OSError, http.client.HTTPException):
            # A hung/broken scorer ends the lease instead of queueing more work.
            self.server.fatal.set()
            self.send_error(502)
        else:
            self.send_json(200, response)
        finally:
            self.server.scoring.release()
