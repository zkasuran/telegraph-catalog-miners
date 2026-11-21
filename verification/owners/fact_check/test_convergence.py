# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import unittest

from convergence_policy import STRATEGIES, parameter_plan, select, training_weight


class ConvergenceTests(unittest.TestCase):
    def test_hard_weights_use_training_scores_only(self):
        self.assertEqual(training_weight("hard_example_24", -.1), 2.)
        self.assertEqual(training_weight("hard_example_24", .1), 1.)
        self.assertEqual(training_weight("low_rate_24", -.1), 1.)
        with self.assertRaises(ValueError):
            training_weight("hard_example_24", float("nan"))

    def test_rate_inventory_and_budget_refuse_invalid_inputs(self):
        self.assertEqual(len(parameter_plan(["classifier", "lora_A"], STRATEGIES[0], 24)), 2)
        for names, epoch in ((["classifier"], 1), (["classifier", "lora_A"], 25)):
            with self.assertRaises(ValueError):
                parameter_plan(names, STRATEGIES[0], epoch)

    def test_complete_validation_only_selection(self):
        reports = [{"strategy": strategy, "epochs_completed": 24, "audit_read": False, "audit_scored": False,
                    "selected_validation": {"val_loss": loss}, "audit_accuracy": audit}
                   for strategy, loss, audit in zip(STRATEGIES, [.5, .4], [1., 0.])]
        self.assertEqual(select(reports), 1)
        with self.assertRaises(ValueError):
            select(reports[:1])
        reports[1]["audit_scored"] = True
        with self.assertRaises(ValueError):
            select(reports)
