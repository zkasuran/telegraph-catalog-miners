# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source and policy controls for a training-only dropout change, no neural work."""
import ast
import copy
import json
import os
import unittest
from unittest.mock import patch

from common import HERE
from dropout_contract import dropout_policy


class DropoutTests(unittest.TestCase):
    def setUp(self):
        self.recipe = json.loads((HERE / "recipe.json").read_text())

    def test_policy_is_zero_and_does_not_change_data_or_gates(self):
        policy = dropout_policy(self.recipe)
        self.assertEqual(policy["lora_dropout"], 0)
        self.assertFalse(policy["data_changed"])
        self.assertFalse(policy["quality_gates_changed"])
        self.assertEqual(self.recipe["lr"], 1e-5)
        self.assertEqual(self.recipe["secondary_training"]["total_rows"], 1008)

    def test_wrong_dropout_or_weakened_control_refused(self):
        for key, value in (("lora_dropout", 0.1), ("lora_dropout", False), ("data_changed", True),
                           ("inference_contract_changed", True), ("native_repeat_control_required", False)):
            recipe = copy.deepcopy(self.recipe)
            recipe["comparison_dropout"][key] = value
            with self.assertRaises(ValueError):
                dropout_policy(recipe)

    def test_actual_model_control_refuses_local_execution(self):
        import train_dropout
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                train_dropout.verify(*([None] * 6))

    def test_actual_training_hyperparameter_reads_the_frozen_policy(self):
        tree = ast.parse((HERE / "run.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "Hyperparams"]
        self.assertEqual(len(calls), 1)
        keyword = next(k for k in calls[0].keywords if k.arg == "lora_dropout")
        self.assertEqual(ast.unparse(keyword.value), "dropout_policy(recipe)['lora_dropout']")


if __name__ == "__main__":
    unittest.main()
