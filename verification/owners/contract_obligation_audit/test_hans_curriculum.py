# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Authored source fixtures only, not model scores or new source annotations."""
import copy
from pathlib import Path
import tempfile
import unittest

from hans_curriculum import preference, require_curriculum, select, shortcuts, source_rows


def fixture(label, premise, hypothesis, identifier):
    return {"gold_label": label, "sentence1": premise, "sentence2": hypothesis, "pairID": identifier,
            "heuristic": "authored", "subcase": "authored", "template": "authored"}


class HansCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.rows = [fixture("entailment", "The doctor smiled.", "The doctor smiled.", "a"),
            fixture("non-entailment", "If the doctor smiled, the lawyer left.", "The doctor smiled.", "b"),
            fixture("entailment", "The banker slept.", "The banker slept.", "c"),
            fixture("non-entailment", "The banker said the lawyer slept.", "The banker slept.", "d")]

    def test_primary_balanced_source_selection_preserves_original_labels(self):
        selected = select(self.rows, 4)
        self.assertEqual({row["pairID"] for row in selected}, {"a", "b", "c", "d"})
        self.assertEqual(selected, select(list(reversed(self.rows)), 4))

    def test_non_entailment_is_not_the_truth_of_not_hypothesis(self):
        row = preference(self.rows[1], "prose")
        self.assertIn("does not establish", row["chosen"])
        self.assertNotIn("It is not true", row["chosen"])
        self.assertEqual(row["original_label"], "non-entailment")

    def test_source_tags_and_class_name_are_not_model_features(self):
        original = preference(self.rows[0], "json")
        changed = {**self.rows[0], "heuristic": "secret label feature", "subcase": "secret subcase", "template": "secret template"}
        for key in ("prompt", "chosen", "rejected"):
            self.assertEqual(original[key], preference(changed, "json")[key])

    def test_source_quote_and_unsupported_performance_boundary_are_retained(self):
        for style in ("json", "prose"):
            row = preference(self.rows[1], style)
            self.assertIn(self.rows[1]["sentence1"], row["chosen"])
            self.assertIn(self.rows[1]["sentence1"], row["rejected"])
            self.assertIn("performance is not assessed", row["chosen"])

    def test_incomplete_hypothesis_or_duplicate_premise_refuses(self):
        with self.assertRaises(ValueError):
            select(self.rows[:1], 2)
        changed = copy.deepcopy(self.rows[:2])
        changed[1]["sentence1"] = changed[0]["sentence1"]
        with self.assertRaises(ValueError):
            select(changed, 2)

    def test_actual_integer_shortcuts_detect_a_stock_label_tell(self):
        rows = [preference(row, "json") for row in self.rows]
        measured, frozen = shortcuts(rows)
        self.assertEqual(set(measured.values()), {.5})
        self.assertEqual(frozen["bow_integer_coefficients"], {})
        for row in rows:
            row["chosen"] += " authored stock winning phrase" * 10
        measured, _ = shortcuts(rows)
        self.assertGreater(measured["answer_only_bow"], .65)
        rows = [preference(row, "json") for row in self.rows]
        for row in rows:
            row["rejected"] += " authored stock rejected phrase" * 10
        measured, frozen = shortcuts(rows)
        self.assertGreater(measured["length"], .65)
        self.assertEqual(frozen["train_fitted_directions"]["length"], -1)

    def test_wrong_original_file_cannot_be_renamed_as_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)/"original-training.jsonl"
            source.write_text('{"authored": "not original data"}')
            with self.assertRaises(ValueError):
                source_rows(source)

    def test_bad_label_or_missing_text_refuses_without_inference(self):
        for changed in ({**self.rows[0], "gold_label": "contradiction"}, {**self.rows[0], "sentence1": None}):
            with self.assertRaises(ValueError):
                preference(changed, "json")

    def test_curriculum_report_is_optional_only_for_non_curriculum_models(self):
        require_curriculum({}, None, None)
        with self.assertRaises(ValueError):
            require_curriculum({"syntactic_curriculum_originals": 320}, None, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
