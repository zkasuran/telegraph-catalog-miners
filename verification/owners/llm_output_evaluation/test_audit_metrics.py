# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Actual pure audit metric source with synthetic rows, never model/audit data."""
import ast
import math
from pathlib import Path
import unittest

from data import RATINGS


SOURCE = (Path(__file__).parent / "runtime_audit.py").read_bytes()
FUNCTION = next(node for node in ast.parse(SOURCE).body if isinstance(node, ast.FunctionDef) and node.name == "metrics")
ENV = {"RATINGS": RATINGS, "math": math}
exec(compile(ast.Module(body=[FUNCTION], type_ignores=[]), "actual pure runtime_audit.metrics", "exec"), ENV)
metrics = ENV["metrics"]


class AuditMetricTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"label": label} for label in RATINGS]
        self.values = [value for _ in RATINGS for value in (2.0, 1.0)]

    def test_every_label_and_count_preserved(self):
        result = metrics(self.rows, self.values)
        self.assertEqual(set(result["rating_slices"]), set(RATINGS))
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(sum(s["n"] for s in result["rating_slices"].values()), len(self.rows))

    def test_missing_label_refused_not_silently_omitted(self):
        with self.assertRaises(ValueError):
            metrics(self.rows[:-1], self.values[:-2])

    def test_unknown_or_nonstring_labels_refused(self):
        for label in ("unknown", None, True, 0):
            with self.subTest(label=label), self.assertRaises(ValueError):
                metrics(self.rows + [{"label": label}], self.values + [2.0, 1.0])

    def test_nonfinite_or_boolean_scores_refused(self):
        for value in (math.nan, math.inf, -math.inf, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                metrics(self.rows, [value] + self.values[1:])

    def test_ties_and_reversals_are_losses(self):
        result = metrics(self.rows, [1.0, 1.0, 0.0, 1.0] + self.values[4:])
        self.assertEqual(result["correct"], len(self.rows) - 2)

    def test_empty_or_wrong_score_count_refused(self):
        for rows, values in (([], []), (self.rows, self.values[:-1])):
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                metrics(rows, values)


if __name__ == "__main__":
    unittest.main()
