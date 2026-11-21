# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Constructed report/encoding failure controls, not model examples or audits."""
from copy import deepcopy
import unittest

from wirecheck import CONTROL_NAMES, literal_root, matrix, validate_output


def report(batches):
    return {"input_roots": [literal_root(batch) for batch in batches],
            "controls": [{"name": name, "passed": True} for name in sorted(CONTROL_NAMES)],
            "neural_operations": False, "replies": "constructed HTTP fixtures, not engine scores"}


class WireReportTests(unittest.TestCase):
    def test_primary_report_shape_smoke(self):
        batches = matrix()
        self.assertTrue(validate_output(report(batches), batches))
        self.assertEqual([len(batch) for batch in batches], [0, 1, 2, 9, 10, 11, 16, 64, 100, 4, 2])

    def test_missing_or_duplicate_control_refused(self):
        batches = matrix()
        for mutation in (lambda v: v["controls"].pop(),
                         lambda v: v["controls"].__setitem__(0, deepcopy(v["controls"][1]))):
            value = report(batches)
            mutation(value)
            with self.assertRaises(ValueError):
                validate_output(value, batches)

    def test_failed_or_numeric_pass_flag_refused(self):
        for flag in (False, 1, "true"):
            batches = matrix()
            value = report(batches)
            value["controls"][0]["passed"] = flag
            with self.assertRaises(ValueError):
                validate_output(value, batches)

    def test_missing_wrong_or_extra_root_refused(self):
        batches = matrix()
        for roots in ([], ["0" * 64] * len(batches), report(batches)["input_roots"] + ["0" * 64]):
            value = report(batches)
            value["input_roots"] = roots
            with self.assertRaises(ValueError):
                validate_output(value, batches)

    def test_neural_or_engine_report_not_relabelled(self):
        batches = matrix()
        for field, item in (("neural_operations", True), ("neural_operations", 0), ("replies", "actual engine scores")):
            value = report(batches)
            value[field] = item
            with self.assertRaises(ValueError):
                validate_output(value, batches)

    def test_extra_fields_do_not_disguise_a_model_audit(self):
        batches = matrix()
        value = report(batches)
        value["quality"] = {"passed": True}
        with self.assertRaises(ValueError):
            validate_output(value, batches)

    def test_int64_boolean_string_and_overflow_controls(self):
        for token in (True, "1", 1.0, 1 << 63, -(1 << 63) - 1):
            with self.assertRaises(ValueError):
                literal_root([[token]])

    def test_batch_order_and_token_order_are_bound(self):
        self.assertNotEqual(literal_root([[1], [2]]), literal_root([[2], [1]]))
        self.assertNotEqual(literal_root([[1, 2]]), literal_root([[2, 1]]))

    def test_reserved_zero_and_empty_vector_are_distinct(self):
        self.assertEqual(literal_root([]), "")
        self.assertNotEqual(literal_root([[]]), literal_root([[0]]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
