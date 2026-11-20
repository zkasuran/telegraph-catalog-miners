# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Actual release-byte binding plus deliberate report failures, without model loading."""
import copy
import unittest

from contracts import HERE, file_sha, package_parts, strict_json, validate_ready


class ReleaseTests(unittest.TestCase):
    def report(self):
        path = HERE / "model/v3-registration-ready.json"
        self.assertEqual(file_sha(path), "fa0cb600acb55f4b8d5c597f1c8f9fe254b5084c66eab7d968037b508e54dbbd")
        return strict_json(path.read_bytes())

    def test_actual_report_package_grants_and_source_bytes_are_bound(self):
        report = validate_ready(self.report())
        package = HERE / "model/SENTIMENT_ANALYSIS-v3.safetensors"
        self.assertEqual(file_sha(package), report["package_sha256"])
        self.assertEqual(package.stat().st_size, report["package_bytes"])
        self.assertEqual(file_sha(HERE / "model/v3-source-manifest.json"), report["source"]["source_manifest_sha256"])
        meta = package_parts(package.read_bytes())[0]["__metadata__"]
        self.assertIn("LicenseRef-zkasuran-SAND-1.0", meta["x-licence"])
        self.assertIn("Apache License", meta["x-goemotions-apache-2.0"])
        self.assertEqual(meta["foundation_hash"], report["foundation_sha256"])
        self.assertEqual(meta["tokenizer_hash"], report["tokenizer_sha256"])
        self.assertEqual(meta["intent"], "SENTIMENT_ANALYSIS")

    def test_actual_release_cannot_hide_missing_format_label_or_low_accuracy(self):
        for kind in ("format", "label", "accuracy", "shortcut"):
            report = copy.deepcopy(self.report())
            if kind == "format":
                report["quality"]["formats"].pop("bare_label")
            elif kind == "label":
                report["quality"]["formats"]["json"]["per_label"].pop("Mixed")
            elif kind == "accuracy":
                report["quality"]["formats"]["bare_label"]["accuracy"] = .79
            else:
                report["quality"]["formats"]["json"]["shortcut_accuracy"] = .9
            with self.assertRaises(ValueError):
                validate_ready(report)

    def test_actual_release_rejects_prefixed_identity_false_runtime_and_nan_bits(self):
        for kind in ("hash", "runtime", "bits"):
            report = copy.deepcopy(self.report())
            if kind == "hash":
                report["package_keccak"] = "0x" + report["package_keccak"]
            elif kind == "runtime":
                report["runtime"]["reference_engine_passed"] = False
            else:
                report["runtime"]["vectors"][0]["score_bits"] = 0x7fc00000
            with self.assertRaises(ValueError):
                validate_ready(report)


if __name__ == "__main__":
    unittest.main()
