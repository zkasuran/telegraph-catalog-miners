# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Literal role and scalar controls; no local model or independent quality test."""
import copy
import json
import math
import os
import unittest
from unittest.mock import patch

from common import HERE
from verdict_contrast import contrast_policy, check_pair, scalar_contrast
from test_source_preferences import population


class VerdictContrastTests(unittest.TestCase):
    def test_actual_training_views_have_same_literal_cross_verdict(self):
        rows, _, _ = population()
        grouped = {}
        for row in rows:
            grouped.setdefault(row["view_id"], []).append(row)
        for pair in grouped.values():
            check_pair(pair)
        first = next(iter(grouped.values()))
        changed = copy.deepcopy(first)
        changed[0]["label"] = changed[1]["label"]
        with self.assertRaises(ValueError):
            check_pair(changed)

    def test_source_order_swap_keeps_same_loss(self):
        c, r = [2., 0.5], [-1., 1.]
        self.assertEqual(scalar_contrast(c, r), scalar_contrast(c[::-1], r[::-1]))

    def test_label_constant_control_cannot_supply_contrast_gap(self):
        # Synthetic score fixtures, not model scores or training material.
        self.assertAlmostEqual(scalar_contrast([2., 0.], [0., 2.]), math.log(2))
        self.assertLess(scalar_contrast([2., 2.], [-2., -2.]), scalar_contrast([2., 0.], [0., 2.]))
        with self.assertRaises(ValueError):
            scalar_contrast([float("nan"), 0], [0, 0])

    def test_policy_cannot_change_data_labels_or_runtime(self):
        recipe = json.loads((HERE / "recipe.json").read_text())
        self.assertEqual(contrast_policy(recipe)["new_rows"], 0)
        for key, value in (("coefficient", True), ("runtime_transform", True), ("source_labels_changed", True), ("new_rows", 2)):
            changed = copy.deepcopy(recipe)
            changed["same_verdict_contrast"][key] = value
            with self.assertRaises(ValueError):
                contrast_policy(changed)

    def test_tensor_contrast_refuses_local_execution(self):
        import train_verdict_contrast
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                train_verdict_contrast.loss(None, None)
            with self.assertRaises(RuntimeError):
                train_verdict_contrast.smoke(None)


if __name__ == "__main__":
    unittest.main()
