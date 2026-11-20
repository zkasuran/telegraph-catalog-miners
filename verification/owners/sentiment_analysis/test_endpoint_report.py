# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Verify actual completed integration report bytes without neural imports."""
import copy
import unittest

from contracts import HERE, file_sha, strict_json, validate_ready
from endpoint_recovery import ORIGIN, RECOVERED, verify_identity


class EndpointReportTests(unittest.TestCase):
    def report(self):
        path = HERE / "model/endpoint-registration-ready.json"
        self.assertEqual(file_sha(path), "d9bd19c9b3e4f959df9ab41c653fd1448f4236c1322d45ca0e78b9636e231cbd")
        return strict_json(path.read_bytes())

    def test_actual_integration_report_and_package_preserve_origin_function(self):
        report = validate_ready(self.report())
        self.assertEqual(file_sha(RECOVERED), report["package_sha256"])
        self.assertEqual(report["run_id"], 44822282420)
        self.assertEqual(report["origin_audit"]["run_id"], 44723258067)
        self.assertTrue(report["origin_audit"]["no_new_fitting"])
        self.assertEqual(verify_identity(ORIGIN.read_bytes(), RECOVERED.read_bytes()), report["byte_preservation"]["tensor_payload_sha256"])

    def test_actual_integration_report_cannot_omit_runtime_or_weaken_quality(self):
        for kind in ("runtime", "quality", "label", "bits"):
            report = copy.deepcopy(self.report())
            if kind == "runtime":
                report["runtime"]["reference_engine_passed"] = False
            elif kind == "quality":
                report["quality"]["formats"]["bare_label"]["accuracy"] = .79
            elif kind == "label":
                report["quality"]["formats"]["json"]["per_label"].pop("Mixed")
            else:
                report["runtime"]["vectors"][0]["score_bits"] = 0x7fc00000
            with self.assertRaises(ValueError):
                validate_ready(report)


if __name__ == "__main__":
    unittest.main()
