# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import json
import unittest

from contracts import finite_bits, response_bits
from registration_common import strict_json


class ContractChecks(unittest.TestCase):
    def test_actual_literal_fields_and_finite_bits(self):
        request = {"batch": [{"tokens": [1, 2]}], "foundation_hash": "aa", "adapter_hash": "bb", "tokenizer_hash": "cc", "input_root": "dd"}
        body = {**request, "results": [{"score": 0x3f800000}]}
        self.assertEqual(response_bits(body, request), [0x3f800000])
        for field in ("foundation_hash", "adapter_hash", "tokenizer_hash"):
            bad = copy.deepcopy(body)
            bad[field] = "0x" + bad[field]
            with self.assertRaisesRegex(ValueError, "literal"):
                response_bits(bad, request)
        body["results"] = []
        with self.assertRaisesRegex(ValueError, "count"):
            response_bits(body, request)

    def test_nonfinite_score_refused(self):
        for value in (True, -1, 1 << 32, 0x7f800000, 0x7fc00000):
            with self.assertRaises(ValueError):
                finite_bits(value)

    def test_duplicate_json_and_nonfinite_refused(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            with self.assertRaises(ValueError):
                strict_json(raw)
        self.assertEqual(strict_json(json.dumps({"x": 1}).encode()), {"x": 1})


if __name__ == "__main__":
    unittest.main()
