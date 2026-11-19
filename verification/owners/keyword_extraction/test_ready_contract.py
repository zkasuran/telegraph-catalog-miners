# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Synthetic schema controls only. These are not source/model evidence or reports."""
import copy
import unittest

from ready_contract import ENGINE, FOUNDATION, TOKENIZER, validate_ready


def synthetic_report():
    metrics = {"accuracy": 0.9, "untrained_accuracy": 0.5, "shortcut_accuracy": 0.5, "per_label": {"synthetic_test_only": 0.9}}
    return {"schema_version": 1, "status": "REGISTRATION_READY", "family": "KEYWORD_EXTRACTION", "run_id": 1,
            "commit": "a" * 40, "serving_commit": "a" * 40, "package_sha256": "b" * 64, "package_keccak": "c" * 64,
            "foundation_sha256": FOUNDATION, "engine_sha256": ENGINE, "tokenizer_sha256": TOKENIZER,
            "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True, "independent_groups": 32, "source_manifest_sha256": "d" * 64},
            "quality": {"passed": True, "audit_pairs": 64, "all_supported_formats_tested": True,
                        "formats": {name: copy.deepcopy(metrics) for name in ("plain", "array", "object")}},
            "runtime": {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
                        "negative_identity_controls": 3, "vectors": [{"tokens": [50281, 1, 50282], "score_bits": 1065353216} for _ in range(8)]}}


class ReadyTests(unittest.TestCase):
    def test_synthetic_shape_and_bare_hash_control(self):
        report = synthetic_report()
        self.assertTrue(validate_ready(report))
        report["package_keccak"] = "0x" + report["package_keccak"]
        with self.assertRaisesRegex(ValueError, "literal bare"):
            validate_ready(report)

    def test_no_missing_audit_parser_or_format(self):
        for field, value in (("audit_pairs", 63), ("passed", False), ("all_supported_formats_tested", False)):
            report = synthetic_report()
            report["quality"][field] = value
            with self.assertRaises(ValueError):
                validate_ready(report)
        report = synthetic_report()
        del report["quality"]["formats"]["object"]
        with self.assertRaisesRegex(ValueError, "every exact"):
            validate_ready(report)
        report = synthetic_report()
        report["runtime"]["full_parser_passed"] = False
        with self.assertRaisesRegex(ValueError, "complete primary"):
            validate_ready(report)

    def test_failed_quality_source_and_finite_bits_controls(self):
        report = synthetic_report()
        report["quality"]["formats"]["plain"]["per_label"] = {"synthetic_test_only": 0.69}
        with self.assertRaises(ValueError):
            validate_ready(report)
        report = synthetic_report()
        report["source"]["audit_used_for_selection"] = True
        with self.assertRaises(ValueError):
            validate_ready(report)
        report = synthetic_report()
        report["runtime"]["vectors"][0]["score_bits"] = 2139095040
        with self.assertRaises(ValueError):
            validate_ready(report)


if __name__ == "__main__":
    unittest.main()
