# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Synthetic report-shape controls, not measured model/audit results."""
import copy
from pathlib import Path
import types
import unittest

from contracts import validate_ready
from test_contracts import synthetic_ready_fixture


class ReadyShapes(unittest.TestCase):
    def malformed(self, dotted, value):
        report = copy.deepcopy(synthetic_ready_fixture())
        fields = dotted.split(".")
        target = report
        for field in fields[:-1]:
            target = target[field]
        target[fields[-1]] = value
        return report

    def test_boolean_evidence_is_not_numeric_proof(self):
        for field, value in (("schema_version", True), ("source.normalized_overlap", False),
                             ("runtime.negative_identity_controls", 3.0)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_ready(self.malformed(field, value))

    def test_all_container_shapes_fail_with_value_error(self):
        for field in ("source", "quality", "quality.formats", "quality.formats.json",
                      "quality.formats.json.per_label", "runtime"):
            for value in (None, [], True, 1, "object"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_ready(self.malformed(field, value))
        report = synthetic_ready_fixture()
        report["runtime"]["vectors"][0] = None
        with self.assertRaises(ValueError):
            validate_ready(report)

    def test_nonstring_hashes_fail_with_value_error(self):
        for field in ("commit", "package_keccak", "package_sha256", "source.source_manifest_sha256"):
            for value in (None, 1, [], {}):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_ready(self.malformed(field, value))

    def test_valid_numbers_and_quality_thresholds_unchanged(self):
        self.assertEqual(validate_ready(synthetic_ready_fixture())["schema_version"], 1)
        validate_ready(self.malformed("source.normalized_overlap", 0.0))
        for field, value in (("quality.formats.json.accuracy", 0.79),
                             ("quality.formats.json.per_label.no", 0.69),
                             ("quality.formats.json.untrained_accuracy", 0.76),
                             ("quality.formats.json.shortcut_accuracy", 0.71)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_ready(self.malformed(field, value))

    def test_boolean_schema_mutation_is_caught(self):
        source = Path(__file__).with_name("contracts.py").read_text()
        mutant = types.ModuleType("mutant_ready_contract")
        exec(compile(source.replace('type(report.get("schema_version")) is not int or ', ''),
                     "mutant_ready_contract", "exec"), mutant.__dict__)
        with self.assertRaises(AssertionError):
            with self.assertRaises(ValueError):
                mutant.validate_ready(self.malformed("schema_version", True))


if __name__ == "__main__":
    unittest.main()
