# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Synthetic source controls. These are not measured neural scores or model data."""
import unittest

from export_record import CHILD, FAMILY, average_ranks, measured_training_record


class ExportRecordTests(unittest.TestCase):
    def test_selected_child_from_previously_measured_pairs(self):
        record = measured_training_record([[1.0, 0.0], [1.0, 0.0]], "a" * 64, audit_used_for_selection=False)
        self.assertEqual(record["family"], FAMILY)
        self.assertEqual(record["children"][0]["intent"], CHILD)
        self.assertEqual(record["children"][0]["pairs"], 2)
        self.assertEqual(record["children"][0]["accuracy"], 1)
        self.assertAlmostEqual(record["children"][0]["spearman"], 1)

    def test_no_fabricated_all_tied_correlation(self):
        with self.assertRaisesRegex(ValueError, "undefined"):
            measured_training_record([[0.0, 0.0]], "a" * 64, audit_used_for_selection=False)

    def test_reversed_preference_is_not_relabelled(self):
        record = measured_training_record([[0.0, 1.0], [0.0, 1.0]], "a" * 64, audit_used_for_selection=False)
        self.assertEqual(record["children"][0]["accuracy"], 0)
        self.assertAlmostEqual(record["children"][0]["spearman"], -1)

    def test_ties_get_average_ranks(self):
        self.assertEqual(average_ranks([2.0, 1.0, 1.0, 3.0]), [3.0, 1.5, 1.5, 4.0])

    def test_identity_and_native_score_refusals(self):
        for pairs, identity in (([], "a" * 64), ([[True, 0.0]], "a" * 64),
                                ([[float("nan"), 0.0]], "a" * 64), ([[1.0]], "a" * 64),
                                ([[1.0, 0.0]], "0x" + "a" * 64), ([[10 ** 400, 0]], "a" * 64)):
            with self.assertRaises(ValueError):
                measured_training_record(pairs, identity, audit_used_for_selection=False)

    def test_audit_is_not_a_selection_split(self):
        for value in (True, 0, None):
            with self.assertRaises(ValueError):
                measured_training_record([[1.0, 0.0]], "a" * 64, audit_used_for_selection=value)


if __name__ == "__main__":
    unittest.main()
