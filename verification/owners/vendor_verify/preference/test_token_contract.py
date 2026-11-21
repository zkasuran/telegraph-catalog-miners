# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Synthetic token evidence controls. No tokenizer or model runs locally."""

import json
import unittest

import token_contract as contract


class TokenContracts(unittest.TestCase):
    def test_primary_full_inventory_smoke(self):
        row = {"prompt": "synthetic input", "chosen": "one", "rejected": "two"}
        calls = []

        def encode(text):
            calls.append(text)
            return [1, 2, 3]

        report = contract.inspect_development(
            {"train": [row] * 2, "validation": [row]}, encode, encode, encode
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["totals"]["texts_checked"], 6)
        self.assertEqual(len(calls), 18)
        self.assertFalse(report["audit_checked"])
        self.assertNotIn(row["prompt"], json.dumps(report))

    def test_retention_and_identity_fail_independently(self):
        vectors = (
            (list(range(513)), list(range(512)), list(range(512)), True, False),
            ([1, 2], [1, 2], [1, 3], False, True),
            ([1, 2], [1], [1], False, False),
        )
        for full, training, node, over, mismatch in vectors:
            report = contract.compare_ids(full, training, node)
            self.assertEqual(report["over_cap"], over)
            self.assertEqual(report["training_node_mismatch"], mismatch)
            self.assertTrue(report["training_truncated"] or report["node_truncated"])

    def test_failure_does_not_skip_remaining_inputs(self):
        rows = [{"prompt": "synthetic", "chosen": "a", "rejected": "b"}] * 2

        def full(text):
            return list(range(513))

        def capped(text):
            return list(range(512))

        report = contract.inspect_development(
            {"train": rows, "validation": rows}, full, capped, capped
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["totals"]["over_cap"], 8)
        self.assertEqual(report["totals"]["training_node_mismatch"], 0)

    def test_audit_and_invalid_vectors_refused(self):
        with self.assertRaises(ValueError):
            contract.inspect_development({"audit": []}, None, None, None)
        for bad in ([], [True], [-1], [1.0], "tokens"):
            with self.assertRaises(ValueError):
                contract.compare_ids(bad, [1], [1])
        boundary = contract.compare_ids(
            list(range(512)), list(range(512)), list(range(512))
        )
        self.assertFalse(boundary["over_cap"])


if __name__ == "__main__":
    unittest.main()
