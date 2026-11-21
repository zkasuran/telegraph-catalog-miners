# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""In-memory synthetic wire controls, never model data or an actual ready report."""
import copy
import json
import unittest

from ready_contract import ENGINE, FOUNDATION, TOKENIZER, decode, validate_wire_shape


class ReadyContractTests(unittest.TestCase):
    def setUp(self):
        # These identities and metrics are labelled synthetic unit controls.
        # They are not persisted as registration-ready.json or submitted anywhere.
        self.expected = {"run_id": 123, "commit": "a" * 40, "package_sha256": "b" * 64,
                         "package_keccak": "c" * 64, "source_manifest_sha256": "d" * 64,
                         "serving_commit": "a" * 40}
        self.control = {"schema_version": 1, "status": "REGISTRATION_READY", "family": "CODE_PATCH_VERIFY",
                        **{k: v for k, v in self.expected.items() if k != "source_manifest_sha256"},
                        "foundation_sha256": FOUNDATION, "engine_sha256": ENGINE, "tokenizer_sha256": TOKENIZER,
                        "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True,
                                   "independent_groups": 32, "source_manifest_sha256": "d" * 64},
                        "quality": {"passed": True, "audit_pairs": 64, "all_supported_formats_tested": True,
                                    "formats": {"json": {"accuracy": 0.9, "untrained_accuracy": 0.5, "shortcut_accuracy": 0.5,
                                                         "per_label": {"true": 0.9, "false": 0.9}}}},
                        "runtime": {"full_parser_passed": True, "reference_engine_passed": True,
                                    "repeat_bits_identical": True, "negative_identity_controls": 3,
                                    "vectors": [{"tokens": [n + 1], "score_bits": 1065353216} for n in range(8)]}}

    def check(self, control=None):
        return validate_wire_shape(json.dumps(control or self.control).encode(), self.expected, ("json",), ("true", "false"))

    def test_primary_literal_shape_control_only(self):
        self.assertEqual(self.check()["run_id"], 123)

    def test_native_identity_types(self):
        for key, value in (("schema_version", True), ("run_id", 123.0), ("package_keccak", "0x" + "c" * 64)):
            control = copy.deepcopy(self.control)
            control[key] = value
            with self.assertRaises(ValueError):
                self.check(control)

    def test_missing_expected_producer_identity(self):
        with self.assertRaises(ValueError):
            validate_wire_shape(json.dumps(self.control).encode(), {}, ("json",), ("true", "false"))

    def test_literal_formats_not_normalized(self):
        control = copy.deepcopy(self.control)
        control["quality"]["formats"]["JSON"] = control["quality"]["formats"].pop("json")
        with self.assertRaises(ValueError):
            self.check(control)

    def test_missing_label(self):
        control = copy.deepcopy(self.control)
        del control["quality"]["formats"]["json"]["per_label"]["false"]
        with self.assertRaises(ValueError):
            self.check(control)

    def test_quality_and_shortcut_controls(self):
        for key, value in (("accuracy", 0.79), ("untrained_accuracy", 0.89), ("shortcut_accuracy", 0.85), ("accuracy", True)):
            control = copy.deepcopy(self.control)
            control["quality"]["formats"]["json"][key] = value
            with self.assertRaises(ValueError):
                self.check(control)
        control = copy.deepcopy(self.control)
        control["quality"]["formats"]["json"]["per_label"]["false"] = 0.69
        with self.assertRaises(ValueError):
            self.check(control)

    def test_independent_source_and_audit_bounds(self):
        for key, value in (("normalized_overlap", False), ("independent_groups", 32.0), ("audit_used_for_selection", True)):
            control = copy.deepcopy(self.control)
            control["source"][key] = value
            with self.assertRaises(ValueError):
                self.check(control)

    def test_tokens_and_finite_bits(self):
        for vector in ({"tokens": [-1], "score_bits": 0}, {"tokens": [True], "score_bits": 0},
                       {"tokens": [1] * 513, "score_bits": 0}, {"tokens": [1], "score_bits": 0x7f800000}):
            control = copy.deepcopy(self.control)
            control["runtime"]["vectors"][0] = vector
            with self.assertRaises(ValueError):
                self.check(control)

    def test_duplicate_reference_vector(self):
        control = copy.deepcopy(self.control)
        control["runtime"]["vectors"][1] = control["runtime"]["vectors"][0]
        with self.assertRaises(ValueError):
            self.check(control)

    def test_parser_and_runtime_flags_are_native(self):
        for key in ("full_parser_passed", "reference_engine_passed", "repeat_bits_identical"):
            control = copy.deepcopy(self.control)
            control["runtime"][key] = 1
            with self.assertRaises(ValueError):
                self.check(control)

    def test_duplicate_nonfinite_and_depth(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e9999}', b"[" * 65 + b"0" + b"]" * 65, b"\xff"):
            with self.assertRaises(ValueError):
                decode(raw)

    def test_unhashable_frozen_names_and_large_accuracy(self):
        with self.assertRaises(ValueError):
            validate_wire_shape(json.dumps(self.control).encode(), self.expected, ([],), ("true", "false"))
        control = copy.deepcopy(self.control)
        control["quality"]["formats"]["json"]["accuracy"] = 10 ** 300
        with self.assertRaises(ValueError):
            self.check(control)


if __name__ == "__main__":
    unittest.main()
