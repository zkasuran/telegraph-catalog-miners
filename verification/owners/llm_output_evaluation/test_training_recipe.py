# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Synthetic source-only recipe controls. No tokenizer, model or fitted probe."""
import copy
import json
from pathlib import Path
import unittest

from data import FORMATS, RATINGS, format_pair, pair, render
from training_recipe import RECIPES, build_recipe, pair_bytes, view_statistics


def originals():
    return [pair({"prompt": f"Synthetic task {index}", "response": f"Synthetic complete output {index}",
                  "helpfulness": index % 5, "group_id": f"{index + 1:064x}",
                  "source_file": "train.jsonl.gz", "source_row": index + 1}) for index in range(160)]


class TrainingRecipeTests(unittest.TestCase):
    def setUp(self):
        self.rows = originals()

    def build(self, name="format-transfer-v1", rows=None):
        return build_recipe(self.rows if rows is None else rows, name, "1" * 64, "2" * 64)

    def test_original_recipe_preserves_every_pair(self):
        views, report = self.build("json-original-v1")
        self.assertEqual(views, self.rows)
        self.assertEqual((report["epochs"], report["training_view_pairs"], report["example_presentations_ceiling"]), (3, 160, 480))

    def test_transfer_covers_formats_without_inventing_groups(self):
        views, report = self.build()
        self.assertEqual(len(views), 335)
        self.assertEqual(report["statistics"]["formats"], {"json": 160, "prose": 160, "bare_label": 15})
        self.assertEqual(report["statistics"]["independent_source_groups"], 160)
        self.assertEqual(set(row["format"] for row in views), set(FORMATS))

    def test_preserves_context_labels_grants_and_complete_outputs(self):
        by_group = {row["group_id"]: row for row in self.rows}
        for view in self.build()[0]:
            original = by_group[view["group_id"]]
            for key in ("prompt", "label", "wrong_label", "upstream_license", "annotation_kind", "human_review_of_conversion"):
                self.assertEqual(view[key], original[key])
            self.assertIn(view["chosen"], render(view["prompt"], view["chosen"]))
            output = json.loads(original["prompt"])["output"]
            if view["format"] == "json":
                self.assertTrue(json.loads(view["chosen"])["assessment"].endswith(output))
            elif view["format"] == "prose":
                self.assertTrue(view["chosen"].endswith(output))

    def test_fixed_balanced_bare_allocation(self):
        views, _ = self.build()
        bare = [row for row in views if row["format"] == "bare_label"]
        for label in RATINGS:
            self.assertEqual(sum(row["label"] == label for row in bare), 3)
            self.assertEqual(sum(row["wrong_label"] == label for row in bare), 3)

    def test_naive_full_expansion_exposes_repetition_failure(self):
        full = [format_pair(row, fmt) for fmt in FORMATS for row in self.rows]
        stats = view_statistics(full)
        self.assertLess(stats["unique_negative_share"], 0.80)
        self.assertGreater(stats["largest_negative_share"], 0.02)

    def test_compute_budget_does_not_grow(self):
        _, report = self.build()
        self.assertEqual(report["epochs"], 1)
        self.assertLessEqual(report["example_presentations_ceiling"], report["original_example_presentations_ceiling"])
        self.assertLessEqual(report["optimizer_steps_ceiling"], report["original_optimizer_steps_ceiling"])
        self.assertFalse(report["statistics"]["surface_fit_run"])
        self.assertFalse(report["statistics"]["full_primary_gate_passed"])

    def test_recipe_is_immutable_and_repeat_bytes_identical(self):
        before = pair_bytes(self.rows)
        first, a = self.build()
        second, b = self.build()
        self.assertEqual(pair_bytes(first), pair_bytes(second))
        self.assertEqual(a, b)
        self.assertEqual(before, pair_bytes(self.rows))

    def test_unknown_recipe_or_bad_frozen_hash_refused(self):
        with self.assertRaises(ValueError):
            self.build("unknown")
        with self.assertRaises(ValueError):
            build_recipe(self.rows, RECIPES[0], "invalid", "2" * 64)

    def test_duplicate_group_or_false_human_review_refused(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["group_id"] = rows[0]["group_id"]
        with self.assertRaises(ValueError):
            self.build(rows=rows)
        rows = copy.deepcopy(self.rows)
        rows[0]["human_review_of_conversion"] = True
        with self.assertRaises(ValueError):
            self.build(rows=rows)

    def test_actual_remote_source_and_workflow_use_the_recipe(self):
        here = Path(__file__).parent
        source = (here / "remote.py").read_text()
        self.assertIn('trainer.gate_dataset(fit_rows, hp)', source)
        self.assertIn('trainer.train(str(base), fit_rows, splits["validation"]', source)
        self.assertIn('epochs=recipe["epochs"]', source)
        self.assertIn('TRAINING_RECIPE', (here / "controller-workflow.yml").read_text())


if __name__ == "__main__":
    unittest.main()
