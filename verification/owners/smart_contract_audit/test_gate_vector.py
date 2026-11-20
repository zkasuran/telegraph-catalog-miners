# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import unittest

from selection import checkpoint_better, checkpoint_key, selection_metrics


def fixture():
    return [{"split": "validation", "response_view": style, "vulnerable": label,
             "facts": {"mechanism": mechanism}} for style in ("prose", "json", "report")
            for label in (True, False) for mechanism in ("access_control", "arithmetic") for _ in range(5)]


class GateVectorChecks(unittest.TestCase):
    def test_scalar_tie_improves_remaining_cells(self):
        rows = fixture()
        earlier = [-1.0 if r["response_view"] == "prose" and r["vulnerable"] is False or i % 5 == 0 else 1.0 for i, r in enumerate(rows)]
        later = [-1.0 if r["response_view"] == "prose" and r["vulnerable"] is False else 1.0 for r in rows]
        a, b = selection_metrics(rows, earlier), selection_metrics(rows, later)
        self.assertEqual(a["selection_score"], b["selection_score"])
        self.assertTrue(checkpoint_better(b, a))
        self.assertFalse(checkpoint_better(a, b))
        self.assertFalse(checkpoint_better(b, b))

    def test_higher_average_cannot_sacrifice_worst_gate(self):
        rows = fixture()
        uneven = [1.0] * len(rows)
        for i, r in enumerate(rows):
            if r["response_view"] == "prose" and r["vulnerable"] is False:
                uneven[i] = -1.0
        balanced = [1.0 if i % 5 else -1.0 for i in range(len(rows))]
        a, b = selection_metrics(rows, uneven), selection_metrics(rows, balanced)
        self.assertGreater(a["aggregate_accuracy"], b["aggregate_accuracy"])
        self.assertFalse(checkpoint_better(a, b))

    def test_missing_margin_audit_flag_or_fake_counts_refused(self):
        rows = fixture()
        good = selection_metrics(rows, [1.0] * len(rows))
        for field, value in (("gate_margins", []), ("audit_used_for_selection", True), ("correct_pairs", True), ("measured_pairs", 0)):
            bad = copy.deepcopy(good)
            bad[field] = value
            with self.assertRaises(ValueError):
                checkpoint_key(bad)
