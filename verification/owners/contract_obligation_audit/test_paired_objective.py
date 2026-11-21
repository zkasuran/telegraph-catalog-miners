# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Authored source/math fixtures only. No local tensor loading or gradients."""
import copy
import math
import sys
import unittest
from unittest.mock import patch

from contracts import FAMILY, sha
from paired_objective import flatten_grouped, grouped_collator, paired_training, require_smoke, scalar_loss
from paired_objective_remote import coupled_training, gradient_smoke, tensor_loss


def fixture(label, text):
    return {"split": "train", "label": label, "evidence": text, "evidence_sha256": sha(text.encode()),
            "hypothesis": "The recipient may disclose the information.", "hypothesis_id": "authored-claim",
            "source_group": "authored-" + label}


class PairedObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.rows = [fixture("Entailment", "Authored permissive source fixture."),
                     fixture("Contradiction", "Authored restrictive source fixture.")]

    def test_primary_source_views_remain_adjacent_and_literal(self):
        grouped = paired_training(self.rows)
        flat = flatten_grouped(grouped)
        self.assertEqual([row["view_style"] for row in flat], ["json", "prose", "json", "prose"])
        for index, source in enumerate(self.rows):
            self.assertEqual(flat[2 * index]["prompt"], source["evidence"])
            self.assertEqual(flat[2 * index + 1]["evidence_sha256"], source["evidence_sha256"])

    def test_actual_collator_adapter_preserves_native_caller_and_validation(self):
        class Recorder:
            def __init__(self, tokenizer, max_length):
                self.tokenizer, self.max_length = tokenizer, max_length
            def __call__(self, rows):
                return rows
        marker = object()
        collator = grouped_collator(Recorder)(marker, 512)
        actual = collator(paired_training(self.rows)[:1])
        self.assertEqual([row["view_style"] for row in actual], ["json", "prose"])
        self.assertIs(collator.original.tokenizer, marker)
        self.assertEqual(collator.original.max_length, 512)
        validation = [{**row, "split": "validation"} for row in self.rows]
        self.assertIs(collator(validation), validation)
        with self.assertRaises(ValueError):
            collator([None])

    def test_audit_or_validation_is_refused_even_if_renamed(self):
        for split in ("audit", "validation"):
            rows = copy.deepcopy(self.rows)
            rows[0]["split"] = split
            with self.assertRaises(ValueError):
                paired_training(rows)

    def test_missing_label_and_duplicate_evidence_are_refused(self):
        with self.assertRaises(ValueError):
            paired_training([self.rows[0], self.rows[0]])
        with self.assertRaises(ValueError):
            paired_training([self.rows[0], fixture("Entailment", "Another authored fixture.")])

    def test_source_or_format_reordering_is_refused(self):
        grouped = paired_training(self.rows)
        grouped[0]["paired_views"].reverse()
        with self.assertRaises(ValueError):
            flatten_grouped(grouped)
        grouped = paired_training(self.rows)
        grouped[0]["paired_views"][1]["chosen"] = "authored tamper"
        with self.assertRaises(ValueError):
            flatten_grouped(grouped)

    def test_oversized_or_mixed_group_batch_is_refused(self):
        grouped = paired_training(self.rows)
        with self.assertRaises(ValueError):
            flatten_grouped([grouped[0]] * 9)
        with self.assertRaises(ValueError):
            flatten_grouped([grouped[0], self.rows[1]])

    def test_stronger_format_cannot_hide_weak_format(self):
        independent = lambda a, b: (math.log1p(math.exp(-a)) + math.log1p(math.exp(-b))) / 2 + .001 * (a*a + b*b) / 2
        self.assertLess(independent(5, -1), independent(4, -1))
        self.assertGreater(scalar_loss(5, -1), scalar_loss(4, -1))
        self.assertLess(scalar_loss(4, 0), scalar_loss(4, -1))
        self.assertEqual(scalar_loss(4, -1), scalar_loss(-1, 4))
        self.assertAlmostEqual(scalar_loss(2, 2), math.log1p(math.exp(-2)) + .004)

    def test_nonfinite_or_unbounded_math_is_refused(self):
        for value in (float("nan"), float("inf"), True, 10001):
            with self.assertRaises(ValueError):
                scalar_loss(value, 0)
        with self.assertRaises(ValueError):
            scalar_loss(0, 0, -1)

    def test_all_tensor_entrypoints_refuse_locally_before_access(self):
        before = set(sys.modules)
        with patch.dict("os.environ", {"OWNER_SOURCE_ONLY": "1"}):
            with self.assertRaisesRegex(RuntimeError, "bounded public GitHub"):
                tensor_loss(object(), object(), object(), .001)
            with self.assertRaisesRegex(RuntimeError, "bounded public GitHub"):
                gradient_smoke(object(), .001)
            with self.assertRaisesRegex(RuntimeError, "bounded public GitHub"):
                with coupled_training(object(), object(), .001):
                    self.fail("local neural path executed")
        self.assertFalse({"torch", "numpy", "transformers", "peft"} & (set(sys.modules) - before))
        require_smoke({"training_objective": "independent"}, None, None)
        with self.assertRaises(ValueError):
            require_smoke({"training_objective": "paired-worst-format"}, None, None)
        freeze = {"training_objective": "paired-worst-format", "family": FAMILY, "run_id": 1,
                  "commit": "authored", "paired_objective_smoke_sha256": "a" * 64}
        report = {"schema_version": 1, "family": FAMILY, "run_id": 1, "commit": "authored",
                  "negative_controls": 2, "status": "PAIRED_OBJECTIVE_TORCH_SMOKE_PASSED",
                  "cases": [{"actual_gradients": [-.5, 0.]} for _ in range(4)]}
        require_smoke(freeze, report, "a" * 64)  # Authored identity contract, not an actual Torch report.
        for key, value in (("run_id", 2), ("negative_controls", 0), ("cases", []), ("cases", [None] * 4)):
            changed = {**report, key: value}
            with self.assertRaises(ValueError):
                require_smoke(freeze, changed, "a" * 64)
        with self.assertRaises(ValueError):
            require_smoke(freeze, report, "b" * 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
