# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import unittest
from pathlib import Path

from contracts import FAMILY, FORMATS
from training_record import development_record, validate_record


class TrainingRecordTests(unittest.TestCase):
    def test_actual_failed_shape_is_refused(self):
        bad = {"family": FAMILY, "trained": True, "audit_required": True,
               "train_sha256": "8546f4a520965eba5a5feda407aaedadd8eda5b7670e5d6bfa21b38be1e5b6bd"}
        with self.assertRaisesRegex(ValueError, "children is empty"):
            validate_record(bad)

    def test_actual_metrics_produce_explicit_development_record(self):
        metrics = {fmt: {"accuracy": 0.5} for fmt in FORMATS}
        record = development_record(metrics)
        self.assertEqual(record["children"][0]["accuracy"], 0.5)
        self.assertEqual(record["children"][0]["pairs"], 64)
        self.assertFalse(record["children"][0]["held_out"])
        self.assertNotIn("spearman", record["children"][0])

    def test_bad_child_count_and_metrics_cannot_be_filled_as_pass(self):
        good = development_record({fmt: {"accuracy": 0.5} for fmt in FORMATS})
        for field, value in (("intent", "OTHER_GENERAL"), ("pairs", True), ("accuracy", float("nan")),
                             ("accuracy", 1.1), ("held_out", "true"), ("spearman", float("inf"))):
            bad = copy.deepcopy(good)
            bad["children"][0][field] = value
            with self.assertRaises(ValueError):
                validate_record(bad)


if __name__ == "__main__":
    unittest.main()
