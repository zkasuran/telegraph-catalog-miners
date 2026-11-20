# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Actual-data minimal-wire controls, not local neural quality measurements."""
import copy
import json
import unittest

from common import HERE, sha
from secondary_contract import validate_secondary, secondary_preferences
from source_preferences import preference_rows
from test_source_preferences import population
from origin_contract import equal_byte_views
from training_weights import weight_policy


class MinimalVerdictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.validation, first = population()
        cls.first_pairs, _ = preference_rows(first, cls.train, cls.validation)
        raw = (HERE / "recovery/locked-secondary.jsonl").read_bytes()
        report = json.loads((HERE / "recovery/locked-secondary-producer.json").read_text())
        cls.views = validate_secondary(raw, report, equal_byte_views(cls.train) + equal_byte_views(cls.validation) + first)

    def test_complete_actual_generation_is_restored_not_rerolled(self):
        raw = (HERE / "recovery/locked-secondary.jsonl").read_bytes()
        producer = json.loads((HERE / "recovery/locked-secondary-producer.json").read_text())
        self.assertEqual(sha(raw), "f4d8c1928e27c666d0255172604e756556696bfc2827d55e852530507f185f96")
        self.assertEqual(producer["run_id"], 44789054038)
        self.assertEqual(producer["same_job_prefix_controls"], 25)

    def test_two_fixed_whole_groups_supply_literal_bare_pairs(self):
        old = copy.deepcopy(self.train + self.first_pairs)
        added, report = secondary_preferences(self.views, old, bare_groups=2)
        self.assertEqual(old, self.train + self.first_pairs)
        bare = [r for r in added if r["style"] == "bare_label"]
        self.assertEqual(len(bare), 4)
        self.assertEqual({r["group_id"] for r in bare}, set(sorted({v["group_id"] for v in self.views})[:2]))
        self.assertEqual(report["new_bare_rows"], 4)
        self.assertEqual(sum(r["style"] == "bare_label" for r in old + added), 16)
        for row in bare:
            self.assertIn(row["chosen"], ("Human", "AI-generated"))
            self.assertIn(row["rejected"], ("Human", "AI-generated"))
            self.assertNotEqual(row["chosen"], row["rejected"])

    def test_extra_exposure_keeps_actual_and_worst_case_repetition_gates(self):
        _, report = secondary_preferences(self.views, self.train + self.first_pairs, bare_groups=2)
        self.assertTrue(report["repetition"]["passed"])
        self.assertLessEqual(report["worst_case_negative_leak"], 0.10)
        self.assertLessEqual(report["repetition"]["top_rejected_share"], 0.02)
        self.assertGreaterEqual(report["repetition"]["unique_rejected_frac"], 0.80)
        with self.assertRaises(ValueError):
            secondary_preferences(self.views, self.train + self.first_pairs, bare_groups=4)

    def test_secondary_weights_are_symmetric_and_not_runtime_scores(self):
        added, _ = secondary_preferences(self.views, self.train + self.first_pairs, bare_groups=2)
        _, report = weight_policy(added, 16)
        self.assertEqual(report["weight_by_origin"]["Human"], report["weight_by_origin"]["AI-generated"])
        self.assertFalse(report["acceptance_thresholds_changed"])
        self.assertEqual(report["bare_weight"], 16)


if __name__ == "__main__":
    unittest.main()
