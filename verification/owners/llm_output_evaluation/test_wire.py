# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Literal public wire contracts and failure controls, no neural runtime locally."""
import copy
import http.client
import threading
import unittest
from common import FOUNDATION, strict_json
from proxy import identity, input_root, validate_request, validate_response, BoundedServer
from wire import verify_qualification_render
from data import FORMATS, format_pair, pair
from test_owner import original


class WireTests(unittest.TestCase):
    def setUp(self):
        self.row = {"keccak": "a" * 64, "runnable_url": "https://tga1.zkasuran.dev/_llm/runnable"}
        self.request = {"batch": [{"tokens": [50281, 22, 50282]}], **identity(self.row)}
        self.request["input_root"] = input_root(self.request["batch"])

    def test_primary_expression(self):
        self.assertTrue(verify_qualification_render()["qualification_template_verified"])

    def test_identity_string_not_normalized(self):
        validate_request(self.request, self.row)
        for key in identity(self.row):
            for variant in ("0x" + self.request[key], self.request[key].upper(), "b" * 64):
                if variant == self.request[key]:
                    continue
                value = {**self.request, key: variant}
                with self.assertRaises(ValueError):
                    validate_request(value, self.row)

    def test_invalid_tokens_and_duplicate_json(self):
        for ids in ([], [True], [-1], [50368], [1] * 513):
            with self.assertRaises(ValueError):
                validate_request({**self.request, "batch": [{"tokens": ids}]}, self.row)
        with self.assertRaises(ValueError):
            strict_json(b'{"batch":[],"batch":[]}')

    def test_response_bits_and_identity(self):
        response = {"results": [{"score": 0x3f800000}], **identity(self.row), "input_root": self.request["input_root"]}
        validate_response(response, self.request, self.row)
        for change in ({"adapter_hash": "0x" + self.row["keccak"]}, {"results": [{"score": True}]},
                       {"results": [{"score": 0x7f800000}]}, {"input_root": "0" * 64}):
            with self.assertRaises(ValueError):
                validate_response({**response, **change}, self.request, self.row)

    def test_actual_http_path_smoke_and_failure(self):
        request = self.request
        row = self.row
        class ExplicitSyntheticBackend:
            calls = 0
            def score(self, value):
                self.calls += 1
                return {"results": [{"score": 0x3f800000}], **identity(row), "input_root": request["input_root"]}
        backend = ExplicitSyntheticBackend()
        server = BoundedServer(("127.0.0.1", 0), row, backend)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            from common import json_bytes
            for value, expected in ((self.request, 200), ({**self.request, "adapter_hash": "0x" + row["keccak"]}, 400),
                                    ({**self.request, "batch": [{"text": "no tokens"}]}, 400)):
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                try:
                    connection.request("POST", "/runnable", json_bytes(value), {"Content-Type": "application/json"})
                    response = connection.getresponse()
                    body = strict_json(response.read())
                    self.assertEqual(response.status, expected)
                    if expected == 200:
                        self.assertEqual(body["adapter_hash"], row["keccak"])
                        self.assertEqual(body["results"], [{"score": 0x3f800000}])
                finally:
                    connection.close()
            self.assertEqual(backend.calls, 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_all_claimed_formats(self):
        row = pair(original())
        for name in FORMATS:
            converted = format_pair(row, name)
            self.assertIn("Excellent", converted["chosen"])
            self.assertIn("Poor", converted["rejected"])


if __name__ == "__main__":
    unittest.main()
