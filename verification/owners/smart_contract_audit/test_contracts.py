# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import json
import unittest

from contracts import finite_bits, response_bits, training_pairs
from registration_common import strict_json


class ContractChecks(unittest.TestCase):
    def test_source_pairs_receive_actual_collator_margin_without_mutation(self):
        rows = [{"prompt": " p\n", "chosen": "yes\r\n", "rejected": " no ", "source_sha256": "source-marker"}]
        before = copy.deepcopy(rows)
        prepared = training_pairs(rows)
        self.assertEqual(prepared[0]["margin"], 0.0)
        for field in ("prompt", "chosen", "rejected", "source_sha256"):
            self.assertEqual(prepared[0][field], rows[0][field])
        self.assertEqual(rows, before)
        self.assertNotIn("margin", rows[0])
        rows[0]["margin"] = 0.25
        self.assertEqual(training_pairs(rows)[0]["margin"], 0.25)

    def test_bad_margin_is_refused_before_model_work(self):
        for value in (True, float("nan"), float("inf"), "0.0"):
            with self.assertRaisesRegex(ValueError, "margin"):
                training_pairs([{"prompt": "p", "chosen": "yes", "rejected": "no", "margin": value}])

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
