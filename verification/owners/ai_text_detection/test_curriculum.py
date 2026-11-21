# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only curriculum population, determinism and refusal controls."""
import copy
import json
import os
import random
import unittest
from unittest.mock import patch

from common import HERE
from curriculum_contract import curriculum_policy, plan, rounds
from test_gradient_diagnostic import training_population


class CurriculumTests(unittest.TestCase):
    def test_actual_unchanged_training_population_and_bare_views(self):
        rows = training_population()
        groups, binding = plan(rows, json.loads((HERE / "recipe.json").read_text()))
        self.assertEqual(binding, json.loads((HERE / "curriculum-plan.json").read_text()))
        self.assertEqual(len(groups), 8)
        self.assertEqual(binding["unique_training_rows"], 1008)
        self.assertEqual(binding["new_source_or_preference_rows"], 0)
        self.assertEqual(binding["presentations_per_epoch"], 64)

    def test_replay_is_balanced_format_isolated_and_does_not_change_global_rng(self):
        groups, binding = plan(training_population(), json.loads((HERE / "recipe.json").read_text()))
        before = random.getstate()
        sequence = rounds(groups, 0, binding["policy"])
        self.assertEqual(before, random.getstate())
        self.assertEqual(sequence, rounds(groups, 0, binding["policy"]))
        for ordered in sequence:
            self.assertEqual({g[0]["view_id"] for g in ordered}, set(binding["view_ids"]))
            for start in (0, 4):
                rows = [r for group in ordered[start:start+4] for r in group]
                self.assertEqual({r["style"] for r in rows}, {"bare_label"})
                self.assertEqual(sum(r["label"] == "Human" for r in rows), 4)
                self.assertEqual(sum(r["label"] == "AI-generated" for r in rows), 4)

    def test_policy_cannot_expand_replays_or_touch_audit_weights_or_runtime(self):
        recipe = json.loads((HERE / "recipe.json").read_text())
        for key, value in (("rounds_per_epoch", 8), ("validation_or_audit_replay", True), ("runtime_transform", True), ("views_per_update", 1)):
            bad = copy.deepcopy(recipe)
            bad["format_curriculum"][key] = value
            with self.assertRaises(ValueError):
                curriculum_policy(bad)
        bad = copy.deepcopy(recipe)
        bad["bare_loss_weight"] = 32
        with self.assertRaises(ValueError):
            curriculum_policy(bad)

    def test_validation_source_role_or_population_corruption_is_refused(self):
        rows = training_population()
        recipe = json.loads((HERE / "recipe.json").read_text())
        for key, value in (("split", "validation"), ("group_id", "f" * 64), ("label", "Mixed")):
            bad = copy.deepcopy(rows)
            bad[0][key] = value
            with self.assertRaises(ValueError):
                plan(bad, recipe)
        with self.assertRaises(ValueError):
            plan(rows[:-2], recipe)

    def test_curriculum_neural_path_refuses_local_execution(self):
        import train_curriculum
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                train_curriculum.replay(*([None] * 12))


if __name__ == "__main__":
    unittest.main()
