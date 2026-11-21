# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Synthetic presentation controls, no model or independent audit data."""
import copy
import json
from pathlib import Path
import unittest

from data import FORMATS, RATINGS, SCORES, format_pair, pair
from presentation import evaluation_view, source_prose_view
from training_recipe import build_recipe, pair_bytes


def originals():
    return [pair({"prompt": f"Synthetic task {index}", "response": f"Complete output {index}: quote \" and newline\nkept intact.",
                  "helpfulness": index % 5, "group_id": f"{index + 1:064x}", "source_file": "train.jsonl.gz",
                  "source_row": index + 1}) for index in range(160)]


class PresentationTests(unittest.TestCase):
    def setUp(self):
        self.rows = originals()

    def test_legacy_recipes_keep_the_exact_old_renderer(self):
        for recipe in ("json-original-v1", "format-transfer-v1", "format-transfer-v2"):
            for name in FORMATS:
                self.assertEqual(evaluation_view(self.rows[0], name, recipe), format_pair(self.rows[0], name))

    def test_json_and_bare_are_not_rewritten(self):
        for name in ("json", "bare_label"):
            self.assertEqual(evaluation_view(self.rows[0], name, "format-transfer-v3"), format_pair(self.rows[0], name))

    def test_labels_scores_context_grants_and_output_are_preserved(self):
        for row in self.rows:
            view = source_prose_view(row)
            for key in ("prompt", "label", "wrong_label", "group_id", "annotation_kind", "upstream_license", "human_review_of_conversion"):
                self.assertEqual(view[key], row[key])
            output = json.loads(row["prompt"])["output"]
            for side, key in (("chosen", "label"), ("rejected", "wrong_label")):
                self.assertIn(row[key], view[side])
                self.assertIn(str(SCORES[RATINGS.index(row[key])]), view[side])
                self.assertTrue(view[side].endswith(output))

    def test_style_assignment_does_not_depend_on_rating(self):
        row = self.rows[0]
        expected = source_prose_view(row)
        for label in RATINGS:
            changed = {**row, "label": label, "wrong_label": RATINGS[(RATINGS.index(label) + 2) % 5]}
            self.assertEqual(source_prose_view(changed)["chosen_presentation"], expected["chosen_presentation"])
            self.assertEqual(source_prose_view(changed)["rejected_presentation"], expected["rejected_presentation"])

    def test_presentation_is_repeatable_and_does_not_mutate_source(self):
        before = pair_bytes(self.rows)
        first = [source_prose_view(row) for row in self.rows]
        second = [source_prose_view(row) for row in self.rows]
        self.assertEqual(pair_bytes(first), pair_bytes(second))
        self.assertEqual(before, pair_bytes(self.rows))

    def test_bad_source_identity_refused(self):
        with self.assertRaises(ValueError):
            source_prose_view({**self.rows[0], "group_id": "bad"})

    def test_v3_keeps_complete_bounded_336_view_training(self):
        views, report = build_recipe(self.rows, "format-transfer-v3", "1" * 64, "2" * 64)
        self.assertEqual(len(views), 336)
        self.assertEqual(report["statistics"]["independent_source_groups"], 160)
        self.assertEqual(report["statistics"]["formats"], {"json": 161, "prose": 160, "bare_label": 15})
        self.assertEqual(report["epochs"], 1)

    def test_actual_audit_source_uses_and_binds_the_versioned_views(self):
        source = (Path(__file__).parent / "runtime_audit.py").read_text()
        self.assertIn('evaluation_view(r, name, recipe["recipe"])', source)
        self.assertIn('audit_rendered_views_sha256', source)


if __name__ == "__main__":
    unittest.main()
