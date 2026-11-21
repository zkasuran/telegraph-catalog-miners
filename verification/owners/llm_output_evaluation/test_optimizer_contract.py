# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Primary source guard counters only, never neural or optimizer execution."""
import ast
import json
from pathlib import Path
import unittest

from data import pair
from optimizer_contract import primary_update_schedule, require_complete_accumulation
from training_recipe import build_recipe


def originals():
    return [pair({"prompt": f"Synthetic task {index}", "response": f"Synthetic complete output {index}",
                  "helpfulness": index % 5, "group_id": f"{index + 1:064x}",
                  "source_file": "train.jsonl.gz", "source_row": index + 1}) for index in range(160)]


class OptimizerInputContractTests(unittest.TestCase):
    def test_actual_primary_guard_exposes_335_view_tail(self):
        report = primary_update_schedule(335, 16, 1)
        self.assertEqual((report["total_updates"], report["covered_batches_per_epoch"], report["unapplied_batches_per_epoch"]), (20, 320, 15))
        self.assertFalse(report["model_or_optimizer_executed"])
        with self.assertRaises(ValueError):
            require_complete_accumulation(335, 1, 16, 1)

    def test_original_recipe_has_complete_accumulation(self):
        self.assertEqual(require_complete_accumulation(160, 1, 16, 3)["total_updates"], 30)

    def test_v2_covers_every_view_within_original_budget(self):
        rows, recipe = build_recipe(originals(), "format-transfer-v2", "1" * 64, "2" * 64)
        self.assertEqual(len(rows), 336)
        self.assertEqual(recipe["statistics"]["formats"], {"json": 161, "prose": 160, "bare_label": 15})
        self.assertEqual(recipe["statistics"]["independent_source_groups"], 160)
        report = require_complete_accumulation(len(rows), 1, 16, recipe["epochs"])
        self.assertEqual((report["total_updates"], report["unapplied_batches_per_epoch"]), (21, 0))
        self.assertLessEqual(recipe["example_presentations_ceiling"], recipe["original_example_presentations_ceiling"])
        original = min(originals(), key=lambda row: row["group_id"])
        supplement = rows[-1]
        for key in ("prompt", "label", "wrong_label", "group_id", "upstream_license", "human_review_of_conversion"):
            self.assertEqual(original[key], supplement[key])
        self.assertEqual(json.loads(original["chosen"]), json.loads(supplement["chosen"]))
        self.assertEqual(json.loads(original["rejected"]), json.loads(supplement["rejected"]))

    def test_unexpected_primary_source_is_refused(self):
        with self.assertRaises(ValueError):
            primary_update_schedule(336, 16, source=b"unreviewed source")

    def test_actual_remote_source_checks_boundary_before_reference(self):
        source = (Path(__file__).parent / "remote.py").read_text()
        train = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "train")
        body = ast.unparse(train)
        self.assertIn("require_complete_accumulation", body)
        self.assertLess(body.index("require_complete_accumulation"), body.index("reference(root)"))
        workflow = (Path(__file__).parent / "controller-workflow.yml").read_text()
        self.assertIn("format-transfer-v2", workflow)


if __name__ == "__main__":
    unittest.main()
