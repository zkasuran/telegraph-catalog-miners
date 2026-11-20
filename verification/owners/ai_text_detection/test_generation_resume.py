# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Real retained-byte contracts and synthetic token controls, no neural work."""
import copy
import json
import unittest

from common import HERE, sha, audit_excerpts
from generation_resume import retained_prefix, generation_policy, check_token_prefix, PREFIX_SHA


class GenerationResumeTests(unittest.TestCase):
    def setUp(self):
        self.raw = (HERE / "recovery/secondary-prefix.jsonl").read_bytes()
        self.producer = json.loads((HERE / "recovery/secondary-prefix-producer.json").read_text())
        self.recipe = json.loads((HERE / "recipe.json").read_text())

    def test_actual_39_successful_groups_are_retained_exactly(self):
        records = retained_prefix(self.raw, self.producer)
        self.assertEqual(len(records), 78)
        self.assertEqual(sha(self.raw), PREFIX_SHA)
        self.assertEqual(self.producer["run_id"], 44784633424)
        self.assertTrue(all(r["split"] == "train" for r in records))

    def test_changed_retained_bytes_or_producer_are_refused(self):
        with self.assertRaises(ValueError):
            retained_prefix(self.raw + b"changed", self.producer)
        for key, value in (("run_id", 1), ("records", 80), ("audit_used", True), ("source_bundle_sha256", "f" * 64)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                retained_prefix(self.raw, {**self.producer, key: value})

    def test_fixed_budget_cannot_change_sources_words_or_audit(self):
        self.assertEqual(generation_policy(self.recipe)["remaining_token_budget"], 128)
        for key, value in (("remaining_token_budget", 256), ("retained_groups", 38), ("word_gate_changed", True),
                           ("source_selection_changed", True), ("audit_used", True), ("same_seed_prefix_control", False),
                           ("retained_groups", 39.0)):
            bad = copy.deepcopy(self.recipe)
            bad["secondary_generation_resume"][key] = value
            with self.assertRaises(ValueError):
                generation_policy(bad)

    def test_actual_short_word_failure_remains_refused(self):
        # Structural arithmetic fixture only, never written as model output.
        with self.assertRaisesRegex(ValueError, "fewer than 24"):
            audit_excerpts(" ".join(["human"] * 24), " ".join(["machine"] * 23))

    def test_token_prefix_control_has_positive_and_negative_cases(self):
        short = list(range(64))
        longer = short + list(range(64, 128))
        check_token_prefix(short, longer)
        wrong = longer.copy()
        wrong[0] = 42
        with self.assertRaises(ValueError):
            check_token_prefix(short, wrong)
        with self.assertRaises(ValueError):
            check_token_prefix(short, longer[:-1])
        wrong = longer.copy()
        wrong[-1] = 50257
        with self.assertRaises(ValueError):
            check_token_prefix(short, wrong)


if __name__ == "__main__":
    unittest.main()
