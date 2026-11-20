# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Synthetic trainer loader-shape failures only, no neural import or model work."""
import copy
import unittest

from scorer import render
from trainer_contract import trainer_pairs


class TrainerContractTests(unittest.TestCase):
    def test_default_margin_keeps_literal_strings_and_source_unchanged(self):
        row = {"prompt": "  α\nTask", "chosen": '["β"]  ', "rejected": '["γ"]', "source": {"test_only": True}}
        before = copy.deepcopy(row)
        adapted = trainer_pairs([row])[0]
        self.assertEqual(adapted["margin"], 0.0)
        self.assertEqual(render(row["prompt"], row["chosen"]), render(adapted["prompt"], adapted["chosen"]))
        self.assertEqual(row, before)
        with self.assertRaises(KeyError):
            _ = row["margin"]  # exact failure in the unadapted primary collator input

    def test_explicit_margin_and_bad_inputs_do_not_change_objective_silently(self):
        row = {"prompt": "Synthetic prompt", "chosen": "Synthetic alpha", "rejected": "Synthetic beta", "margin": 0.25}
        self.assertEqual(trainer_pairs([row])[0]["margin"], 0.25)
        for value in (True, "0", float("nan"), float("inf"), None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                trainer_pairs([{**row, "margin": value}])
        with self.assertRaises(ValueError):
            trainer_pairs([{**row, "chosen": ""}])


if __name__ == "__main__":
    unittest.main()
