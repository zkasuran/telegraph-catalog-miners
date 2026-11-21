# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Authored HTTP fixtures. No neural process is created or used by these tests."""
import http.client
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock, patch

from audit_remote import request
from contracts import FOUNDATION
from executor import ReferenceExecutor


def wire():
    return {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"], "adapter_hash": "a" * 64,
            "tokenizer_hash": FOUNDATION["tokenizer.json"], "results": [{"score": 0}]}


def backend(port=1):
    value = ReferenceExecutor.__new__(ReferenceExecutor)
    value.package_keccak, value.port = "a" * 64, port
    value.process = Mock()
    value.process.poll.return_value = None
    value.lock = threading.Lock()
    return value


class DeadlineTests(unittest.TestCase):
    def serve(self, drip):
        class Fixture(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def do_POST(self):
                raw = json.dumps(wire(), separators=(",", ":")).encode()
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    for index in range(0, len(raw), 8 if drip else len(raw)):
                        self.wfile.write(raw[index:index + (8 if drip else len(raw))])
                        self.wfile.flush()
                        if drip:
                            time.sleep(.015)
                except OSError:
                    pass
            def log_message(self, *_):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
        server.daemon_threads = True
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(worker.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return backend(server.server_port)

    def test_healthy_actual_http_client_keeps_literal_contract(self):
        value = self.serve(False)
        self.assertEqual(value.score(request([[1, 2]], "a" * 64), timeout=2), wire())
        self.assertFalse(value.lock.locked())

    def test_dripped_response_cannot_extend_total_request_budget(self):
        value = self.serve(True)
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            value.score(request([[1, 2]], "a" * 64), timeout=.06)
        self.assertLess(time.monotonic() - started, .4)
        self.assertFalse(value.lock.locked())

    def test_connection_construction_failure_releases_scorer_lock(self):
        value = backend()
        with patch("executor.http.client.HTTPConnection", side_effect=OSError("authored constructor refusal")):
            with self.assertRaises(OSError):
                value.score(request([[1]], "a" * 64))
        self.assertFalse(value.lock.locked())

    def test_lock_wait_is_deducted_from_http_budget(self):
        value = backend()
        clock = [100.0]
        value.lock = Mock()
        def acquire(*, timeout):
            clock[0] += 2.0
            return True
        value.lock.acquire.side_effect = acquire
        response = Mock(status=200)
        response.read.return_value = json.dumps(wire()).encode()
        connection = Mock()
        connection.getresponse.return_value = response
        connection.sock = None
        with patch("executor.time.monotonic", side_effect=lambda: clock[0]), patch("executor.http.client.HTTPConnection", return_value=connection) as constructor:
            self.assertEqual(value.score(request([[1]], "a" * 64), timeout=5), wire())
        self.assertEqual(constructor.call_args.kwargs["timeout"], 3.0)
        value.lock.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
