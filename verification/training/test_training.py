# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import unittest
from unittest.mock import patch

from run import remote_only, verdict
from runtime_check import decode_bits


class TrainingTests(unittest.TestCase):
    def report(self):
        return {"dataset_gate": {"ok": True}, "formats": {name: {
            "pairwise_accuracy": 0.85, "beats_length_baseline": True,
            "accuracy_by_label": {"Positive": 0.9, "Negative": 0.8}}
            for name in ("json", "prose", "bare_label")},
            "untrained": {name: {"pairwise_accuracy": 0.5} for name in ("json", "prose", "bare_label")}}

    def test_good_result_passes(self):
        self.assertEqual(verdict(self.report())["verdict"], "PASS")

    def test_aggregate_cannot_hide_a_failed_label(self):
        report = self.report()
        report["formats"]["json"]["accuracy_by_label"]["Negative"] = 0.6
        self.assertEqual(verdict(report)["verdict"], "FAIL")

    def test_bad_format_is_not_averaged_away(self):
        report = self.report()
        report["formats"]["bare_label"] = {**report["formats"]["json"], "pairwise_accuracy": 0.6}
        report["untrained"]["bare_label"] = {"pairwise_accuracy": 0.5}
        self.assertEqual(verdict(report)["verdict"], "FAIL")

    def test_no_improvement_and_dataset_failure_are_refused(self):
        report = self.report()
        report["untrained"]["json"]["pairwise_accuracy"] = 0.83
        self.assertEqual(verdict(report)["verdict"], "FAIL")
        report = self.report()
        report["dataset_gate"]["ok"] = False
        self.assertEqual(verdict(report)["verdict"], "FAIL")

    def test_empty_evaluation_cannot_pass(self):
        report = self.report()
        report["formats"] = {}
        self.assertEqual(verdict(report)["verdict"], "FAIL")

    def test_local_training_is_refused_before_importing_torch(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(RuntimeError, "only in the bounded"):
            remote_only()

    def test_nonfinite_or_invalid_score_bits_are_refused(self):
        self.assertEqual(decode_bits(0x3f800000), 1.0)
        for bits in (True, -1, 1 << 32, 0x7fc00000, 0x7f800000):
            with self.assertRaises(ValueError):
                decode_bits(bits)


if __name__ == "__main__":
    unittest.main()
