# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Synthetic token evidence controls. No tokenizer or model runs locally."""

import copy
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

    def test_training_filter_primary_smoke_preserves_order_and_source(self):
        rows = [
            {
                "prompt": "synthetic",
                "chosen": "a",
                "rejected": "b",
                "group_id": f"{i:064x}",
            }
            for i in range(65)
        ]
        rows[-1]["chosen"] = "oversize"
        before = json.dumps(rows)

        def full(text):
            return list(range(513 if "oversize" in text else 512))

        kept, encoded, report = contract.filter_training(rows, full)
        self.assertEqual(kept, rows[:64])
        self.assertEqual(before, json.dumps(rows))
        self.assertEqual(report["retained_pairs"], 64)
        self.assertEqual(len(report["excluded"]), 1)
        self.assertEqual(report["training_sha256"], contract.data.sha(encoded))
        self.assertNotIn("oversize", json.dumps(report))
        self.assertFalse(report["audit_checked"])

    def test_filter_refuses_invalid_vectors_and_coverage_loss(self):
        rows = [
            {
                "prompt": "synthetic",
                "chosen": "a",
                "rejected": "b",
                "group_id": f"{i:064x}",
            }
            for i in range(80)
        ]
        with self.assertRaises(ValueError):
            contract.filter_training(rows, lambda text: [True])
        with self.assertRaises(ValueError):
            contract.filter_training(rows, lambda text: list(range(513)))
        with self.assertRaises(ValueError):
            contract.filter_training(rows[:63], lambda text: [1])
        with self.assertRaises(ValueError):
            contract.filter_training([rows[0]] * 80, lambda text: [1])

    def test_capacity_repair_does_not_allow_identity_or_validation_loss(self):
        row = {"prompt": "synthetic", "chosen": "a", "rejected": "b"}
        report = contract.inspect_development(
            {"train": [row], "validation": [row]},
            lambda text: [1],
            lambda text: [1],
            lambda text: [1],
        )
        self.assertTrue(contract.require_training_capacity_only(report))
        for split, cause in (
            ("validation", "over_cap"),
            ("validation", "node_truncated"),
            ("train", "training_node_mismatch"),
            ("train", "training_truncated"),
        ):
            bad = copy.deepcopy(report)
            bad["splits"][split]["counts"][cause] = 1
            with self.assertRaises(ValueError):
                contract.require_training_capacity_only(bad)
        capacity = copy.deepcopy(report)
        for cause in ("over_cap", "training_truncated", "node_truncated"):
            capacity["splits"]["train"]["counts"][cause] = 1
        self.assertTrue(contract.require_training_capacity_only(capacity))


if __name__ == "__main__":
    unittest.main()
