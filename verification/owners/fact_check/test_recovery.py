# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import unittest

from recovery_policy import STRATEGIES, parameter_plan, recovery_selection


class RecoveryTests(unittest.TestCase):
    def test_distinct_freezing_and_learning_rates(self):
        names = ["model.lora_A.weight", "classifier.modules_to_save.weight"]
        self.assertEqual(len(parameter_plan(names, "head_only", 12)), 1)
        self.assertEqual(len(parameter_plan(names, "head_first", 3)), 1)
        groups = parameter_plan(names, "head_first", 4)
        self.assertEqual(len(groups), 2)
        self.assertEqual([group["lr"] for group in groups], [.0005, .00001])
        self.assertEqual(groups[0]["weight_decay"], 0.)

    def test_unknown_inventory_strategy_and_budget_fail_closed(self):
        for names, strategy, epoch in ((["unknown"], "head_only", 1), (["classifier"], "head_only", 1),
                                       (["lora_A", "classifier"], "guess", 1), (["lora_A", "classifier"], "head_only", 13)):
            with self.assertRaises(ValueError):
                parameter_plan(names, strategy, epoch)

    def test_validation_only_selection_and_complete_ablation_control(self):
        reports = [{"strategy": strategy, "epochs_completed": 12, "audit_read": False, "audit_scored": False,
                    "selected_validation": {"val_loss": loss}, "audit_accuracy": audit}
                   for strategy, loss, audit in zip(STRATEGIES, [.7, .4, .6], [1., 0., 1.])]
        self.assertEqual(recovery_selection(reports), 1)
        with self.assertRaises(ValueError):
            recovery_selection(reports[:2])
        reports[1]["audit_read"] = True
        with self.assertRaises(ValueError):
            recovery_selection(reports)
