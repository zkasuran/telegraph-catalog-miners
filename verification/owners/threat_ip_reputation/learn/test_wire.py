# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Literal wire smoke tests with a labelled non-neural fake backend."""
import copy
import json
import threading
import unittest
import urllib.error
import urllib.request

from contracts import FOUNDATION
from wire import Server, input_root, validate_request, validate_response


class FakeBackend:
    def __init__(self, identity):
        self.identity = identity
        self.calls = 0

    def score(self, value):
        self.calls += 1
        return {**self.identity, "results": [{"score": 0x3f800000} for item in value["batch"]],
                **({"input_root": input_root(value["batch"])} if value.get("input_root") else {})}


class WireTests(unittest.TestCase):
    def setUp(self):
        self.identity = {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
                         "tokenizer_hash": FOUNDATION["tokenizer.json"], "adapter_hash": "1" * 64}
        self.backend = FakeBackend(self.identity)
        self.server = Server(("127.0.0.1", 0), self.identity, self.backend)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)

    def post(self, value):
        request = urllib.request.Request("http://127.0.0.1:" + str(self.server.server_address[1]) + "/runnable",
                                         data=json.dumps(value).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_actual_http_primary_fake_path_preserves_literal_bits(self):
        value = {**self.identity, "batch": [{"tokens": [50281, 22, 50282]}]}
        value["input_root"] = input_root(value["batch"])
        code, result = self.post(value)
        self.assertEqual(code, 200)
        self.assertEqual(result, {**self.identity, "input_root": value["input_root"], "results": [{"score": 0x3f800000}]})
        self.assertEqual(self.backend.calls, 1)

    def test_three_wrong_identities_are_refused_before_backend(self):
        value = {**self.identity, "batch": []}
        for field in self.identity:
            code, result = self.post({**value, field: "0" * 64})
            self.assertEqual(code, 400)
            self.assertEqual(result["error"], "invalid_request")
        self.assertEqual(self.backend.calls, 0)
        for field in self.identity:
            self.assertEqual(self.post({**value, field: "0x" + value[field]})[0], 400)

    def test_token_root_types_and_finite_reply_controls(self):
        value = {**self.identity, "batch": [{"tokens": [50281, 22, 50282]}]}
        for tokens in ([], [True], [50368], [1] * 513):
            self.assertEqual(self.post({**value, "batch": [{"tokens": tokens}]})[0], 400)
        self.assertEqual(self.post({**value, "input_root": "0" * 64})[0], 400)
        for bits in (True, -1, 1 << 32, 0x7f800000, 0x7fc00000):
            with self.assertRaises(ValueError):
                validate_response({**self.identity, "results": [{"score": bits}]}, value, self.identity)
        with self.assertRaises(ValueError):
            validate_response({**self.identity, "foundation_hash": "0x" + self.identity["foundation_hash"], "results": [{"score": 0}]}, value, self.identity)


if __name__ == "__main__":
    unittest.main()
