# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pure scalar/policy controls for an additional train-only consistency loss."""
import copy
import json
import os
import unittest
from unittest.mock import patch

from common import HERE
from source_margin import source_margin_policy, scalar_target, scalar_consistency


class SourceMarginTests(unittest.TestCase):
    def setUp(self):
        self.recipe = json.loads((HERE / "recipe.json").read_text())
        self.policy = source_margin_policy(self.recipe)

    def test_recorded_origin_swap_has_identical_target_and_loss(self):
        for value in (-1000., -2., 0., 2., 1000.):
            self.assertEqual(scalar_target(value, 1, self.policy), scalar_target(-value, 0, self.policy))
            self.assertEqual(scalar_consistency(0, 0, value, 1, self.policy), scalar_consistency(0, 0, -value, 0, self.policy))

    def test_wrong_teacher_never_reverses_source_preference(self):
        for value in (-1000., -2., 0., 2., 1000.):
            target = scalar_target(value, 1, self.policy)
            self.assertGreaterEqual(target, 0.25)
            self.assertLessEqual(target, 6)
            self.assertLess(scalar_consistency(target, 0, value, 1, self.policy), scalar_consistency(-target, 0, value, 1, self.policy))

    def test_policy_cannot_change_labels_detachment_or_runtime(self):
        for key, value in (("teacher_detached", False), ("source_labels_unchanged", False), ("runtime_transform", True),
                           ("audit_used", True), ("coefficient", True), ("coefficient", 1)):
            changed = copy.deepcopy(self.recipe)
            changed["source_margin_consistency"][key] = value
            with self.assertRaises(ValueError):
                source_margin_policy(changed)

    def test_nonfinite_and_wrong_role_controls_refused(self):
        with self.assertRaises(ValueError):
            scalar_target(float("nan"), 1, self.policy)
        with self.assertRaises(ValueError):
            scalar_target(0, True, self.policy)
        with self.assertRaises(ValueError):
            scalar_consistency(float("inf"), 0, 0, 1, self.policy)

    def test_tensor_path_refuses_local_execution(self):
        import train_source_margin
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                train_source_margin.consistency(None, None, None, None, None)
            with self.assertRaises(RuntimeError):
                train_source_margin.smoke(None, None)


if __name__ == "__main__":
    unittest.main()
