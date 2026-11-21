# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pure source/identity/policy/guard checks. Neural head fitting is remote only."""
import ast
import copy
import json
import os
import unittest
from unittest.mock import patch

from common import HERE
from headfit_contract import headfit_policy, effective_recipe, plan, check_start
from test_gradient_diagnostic import training_population


class HeadFitTests(unittest.TestCase):
    def test_actual_complete_training_and_improved_starting_state(self):
        recipe = json.loads((HERE / "recipe.json").read_text())
        binding = plan(training_population(), recipe)
        self.assertEqual(binding, json.loads((HERE / "headfit-plan.json").read_text()))
        self.assertEqual(binding["training_rows"], 1008)
        self.assertEqual(binding["source_view_groups"], 504)
        self.assertEqual(binding["new_data"], 0)
        active = effective_recipe(recipe)
        checked = check_start((HERE / active["checkpoint_file"]).read_bytes(), recipe)
        self.assertEqual(checked["producer_run_id"], 44860695142)
        self.assertEqual(checked["selected_epoch"], 3)
        self.assertNotEqual(active["checkpoint_sha256"], recipe["checkpoint_sha256"])

    def test_policy_cannot_fit_encoder_readout_or_use_audit(self):
        recipe = json.loads((HERE / "recipe.json").read_text())
        for key, value in (("encoder_frozen", False), ("origin_readout_frozen", False), ("trainable_tensors", 178), ("steps", 1000),
                           ("audit_used_for_selection", True), ("runtime_transform", True), ("checkpoint_sha256", "0"*64)):
            bad = copy.deepcopy(recipe)
            bad["head_fit"][key] = value
            with self.assertRaises(ValueError):
                headfit_policy(bad)

    def test_held_out_or_nonzero_margin_training_is_refused(self):
        recipe = json.loads((HERE / "recipe.json").read_text())
        rows = training_population()
        for key, value in (("split", "validation"), ("margin", 1), ("group_id", "f"*64)):
            bad = copy.deepcopy(rows)
            bad[0][key] = value
            with self.assertRaises(ValueError):
                plan(bad, recipe)

    def test_head_fit_and_tensor_objective_refuse_local_neural_work(self):
        import train_headfit
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                train_headfit.fit(*([None]*9))
            with self.assertRaises(RuntimeError):
                train_headfit.objective(*([None]*7))

    def test_feature_hook_is_observation_only_and_removed_in_finally(self):
        tree = ast.parse((HERE / "train_headfit.py").read_text())
        observer = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "observe")
        self.assertFalse(any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(observer)))
        fit = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fit")
        self.assertTrue(any("hook.remove()" in ast.unparse(n) for n in fit.body if isinstance(n, ast.Try)))


if __name__ == "__main__":
    unittest.main()
