# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import unittest

from development_controls import (
    RIDGE_GRID,
    aggregate,
    qualification_development_gate,
    select_candidate,
    variants,
)


def rows():
    return [
        {
            "prompt": "synthetic first",
            "chosen": "long answer",
            "rejected": "short",
            "source_preferred_response": 1,
            "source_strength": -2,
        },
        {
            "prompt": "synthetic second",
            "chosen": "yes",
            "rejected": "other answer",
            "source_preferred_response": 2,
            "source_strength": 1,
        },
    ]


class DevelopmentControls(unittest.TestCase):
    def test_neural_entry_refuses_local_before_import(self):
        import sys
        from unittest.mock import patch
        from runtime_head import train_frozen_head

        with patch.dict("os.environ", {"GITHUB_ACTIONS": "false"}):
            with self.assertRaises(RuntimeError):
                train_frozen_head(None, None, None, None, None, None, None, None, None)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_literal_rows_not_normalized(self):
        original = rows()
        self.assertIs(variants(original)["literal"], original)
        self.assertEqual(
            variants(original)["displaced_prompt"][0]["prompt"], original[1]["prompt"]
        )
        self.assertEqual(original[0]["prompt"], "synthetic first")

    def test_neutral_control_is_not_new_truth(self):
        self.assertEqual(variants(rows())["answer_only"][0]["prompt"], "")
        self.assertEqual(variants(rows())["prompt_only"][0]["chosen"], "")

    def test_exact_zero_ties_are_not_wins(self):
        result = aggregate(rows(), [0.0, 1.0])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["ties"], 1)
        self.assertEqual(result["preferred_response"]["1"]["accuracy"], 0)

    def test_negative_and_label_coverage(self):
        result = aggregate(rows(), [1.0, -1.0])
        self.assertEqual(result["length_direction"]["chosen_longer"]["accuracy"], 1)
        self.assertEqual(result["source_strength"]["less_than_two"]["accuracy"], 0)

    def test_incomplete_and_hostile_scores_refuse(self):
        for values in ([1.0], [True, 1.0], [float("nan"), 1.0], [float("inf"), 1.0]):
            with self.assertRaises(ValueError):
                aggregate(rows(), values)

    def test_complete_predeclared_grid(self):
        candidates = [
            {"ridge": r, "validation": {"pairs": 2, "accuracy": 0.5}}
            for r in RIDGE_GRID
        ]
        self.assertEqual(select_candidate(candidates), 3)
        for bad in (candidates[:-1], list(reversed(candidates))):
            with self.assertRaises(ValueError):
                select_candidate(bad)

    def test_gate_cannot_be_shortcut_or_missing_label(self):
        metric = aggregate(rows(), [1.0, 1.0])
        self.assertTrue(qualification_development_gate(metric, 0.5, 0.6))
        self.assertFalse(qualification_development_gate(metric, 0.99, 0.6))
        self.assertFalse(qualification_development_gate(metric, 0.5, 0.99))
        metric["preferred_response"]["2"]["pairs"] = 0
        self.assertFalse(qualification_development_gate(metric, 0.5, 0.6))


if __name__ == "__main__":
    unittest.main()
