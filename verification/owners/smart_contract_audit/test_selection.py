# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import unittest
from unittest.mock import patch

from selection import FORMATS, selection_metrics
from trainer_selection import adapt_source, install


def fixture():
    return [{"split": "validation", "response_view": name, "vulnerable": value,
             "facts": {"mechanism": mechanism}} for name in FORMATS for value in (True, False)
            for mechanism in ("access_control", "arithmetic") for _ in range(5)]


class SelectionChecks(unittest.TestCase):
    def test_primary_selection_spares_balanced_lower_average_candidate(self):
        rows = fixture()
        uneven = [1.0] * len(rows)
        for i, row in enumerate(rows):
            if row["response_view"] == "prose" and row["vulnerable"] is False:
                uneven[i] = -1.0
        balanced = [1.0 if i % 5 else -1.0 for i in range(len(rows))]
        a, b = selection_metrics(rows, uneven), selection_metrics(rows, balanced)
        self.assertGreater(a["aggregate_accuracy"], b["aggregate_accuracy"])
        self.assertLess(a["selection_score"], b["selection_score"])
        self.assertAlmostEqual(b["selection_score"], 0.0)
        self.assertFalse(b["registration_ready"])

    def test_actual_difference_alignment_and_ties_are_not_wins(self):
        rows = fixture()
        result = selection_metrics(rows, [0.0] * len(rows))
        self.assertEqual(result["aggregate_accuracy"], 0.0)
        self.assertEqual(result["selection_score"], -0.8)
        with self.assertRaises(ValueError):
            selection_metrics(rows, [1.0])

    def test_audit_train_missing_cells_and_nonfinite_are_refused(self):
        rows = fixture()
        for split in ("audit", "train"):
            changed = copy.deepcopy(rows)
            changed[0]["split"] = split
            with self.assertRaisesRegex(ValueError, "validation only"):
                selection_metrics(changed, [1.0] * len(changed))
        changed = [r for r in rows if r["response_view"] != "report"]
        with self.assertRaisesRegex(ValueError, "missing"):
            selection_metrics(changed, [1.0] * len(changed))
        for value in (True, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                selection_metrics(rows, [value] * len(rows))

    def test_source_adaptation_and_execution_refuse_wrong_identity_and_local_model_work(self):
        with self.assertRaisesRegex(ValueError, "pinned"):
            adapt_source(b"not the pinned trainer")
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Model loading"):
                install("unused-source-path", "unused-output-path")


if __name__ == "__main__":
    unittest.main()
