# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import json
import unittest

from common import FOUNDATION, strict_json
from contracts import finite_bits, literal_reply, orientation, quality_gate, read_splits, ready_report, source_gate
from licensing import tensor_parts


class Contracts(unittest.TestCase):
    def test_original_data_and_group_counts(self):
        recipe, _, splits = read_splits()
        self.assertEqual([len(splits[key]) for key in ("train", "validation", "audit")], [256, 64, 128])
        self.assertEqual(recipe["files"]["audit"]["pages"], 39)
        self.assertEqual(source_gate()["audit_metrics"], "sealed")

    def test_orientation_exact_not_suffix(self):
        row = {"prompt": json.dumps({"query": "Which Python version added za?"}), "facts": [{"entity": "a"}, {"entity": "za"}]}
        self.assertEqual(orientation(row), "target_second")

    def test_unchanged_quality_floors(self):
        measured = {"accuracy": .80, "per_label": {"first": .70, "second": .90}}
        self.assertTrue(quality_gate(measured, .70, .60))
        for field, changed in (("accuracy", .79), ("per_label", {"first": .69})):
            bad = copy.deepcopy(measured)
            bad[field] = changed
            self.assertFalse(quality_gate(bad, .50, .50))
        self.assertFalse(quality_gate(measured, .76, .50))
        self.assertFalse(quality_gate(measured, .50, .71))

    def test_literal_hash_echo_not_normalized(self):
        reply = {"adapter_hash": "a" * 64, "foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
                 "tokenizer_hash": FOUNDATION["tokenizer.json"], "results": [{"score": 1065353216}]}
        self.assertEqual(literal_reply(reply, 1, "a" * 64), [1065353216])
        for value in ("0x" + "a" * 64, "A" * 64, "a" * 64 + " "):
            changed = copy.deepcopy(reply)
            changed["adapter_hash"] = value
            with self.assertRaises(ValueError):
                literal_reply(changed, 1, "a" * 64)

    def test_nonfinite_bool_and_count_refused(self):
        for value in (True, -1, 1 << 32, 0x7fc00000, 0x7f800000):
            with self.assertRaises(ValueError):
                finite_bits(value)

    def test_duplicate_wire_field_refused(self):
        with self.assertRaises(ValueError):
            strict_json(b'{"adapter_hash":"a","adapter_hash":"b"}')

    def test_checkpoint_header_malformed_is_not_loaded(self):
        for raw in (b"", b"\0" * 8, b"\xff" * 8):
            with self.assertRaises(ValueError):
                tensor_parts(raw)


class ReadyReportContracts(unittest.TestCase):
    """Structural fixtures only. These numbers are not measured model results."""

    def setUp(self):
        recipe, _, _ = read_splits()
        self.report = {"schema_version": 1, "status": "REGISTRATION_READY", "family": "WEB_SEARCH",
                       "run_id": 1, "commit": "a" * 40, "serving_commit": "a" * 40,
                       "package_sha256": "b" * 64, "package_keccak": "c" * 64, "package_bytes": 1024,
                       "foundation_sha256": FOUNDATION["foundation_lora_fp32.onnx"],
                       "engine_sha256": FOUNDATION["foundation_engine.wasm"], "tokenizer_sha256": FOUNDATION["tokenizer.json"],
                       "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True,
                                  "source_manifest_sha256": recipe["source_manifest_sha256"], "independent_groups": 39},
                       "quality": {"passed": True, "audit_pairs": 128, "all_supported_formats_tested": True,
                                   "formats": {"raw_json": {"accuracy": .85, "per_label": {"target_first": .8, "target_second": .9},
                                                           "untrained_accuracy": .5, "shortcut_accuracy": .5}}},
                       "runtime": {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
                                   "negative_identity_controls": 3, "vectors": [{"tokens": [1, 2], "score_bits": 1065353216} for _ in range(8)]}}

    def test_structural_fixture_is_accepted_not_an_audit(self):
        self.assertIs(ready_report(self.report), self.report)

    def test_missing_run_commit_package_identity_refused(self):
        for key in ("run_id", "commit", "serving_commit", "package_sha256", "package_keccak", "package_bytes"):
            with self.subTest(key=key):
                bad = copy.deepcopy(self.report)
                del bad[key]
                with self.assertRaises(ValueError):
                    ready_report(bad)

    def test_run_and_count_booleans_refused(self):
        for key, value in (("run_id", True), ("run_id", 0), ("package_bytes", True), ("package_bytes", 64 << 20 | 1)):
            with self.subTest(key=key, value=value):
                bad = copy.deepcopy(self.report)
                bad[key] = value
                with self.assertRaises(ValueError):
                    ready_report(bad)
        for section, key in (("source", "independent_groups"), ("quality", "audit_pairs"), ("runtime", "negative_identity_controls")):
            bad = copy.deepcopy(self.report)
            bad[section][key] = True
            with self.assertRaises(ValueError):
                ready_report(bad)

    def test_literal_hash_and_commit_variants_refused(self):
        for key in ("commit", "serving_commit", "package_sha256", "package_keccak"):
            for change in (lambda text: "0x" + text, str.upper, lambda text: text + " "):
                with self.subTest(key=key, change=change):
                    bad = copy.deepcopy(self.report)
                    bad[key] = change(bad[key])
                    with self.assertRaises(ValueError):
                        ready_report(bad)

    def test_source_manifest_identity_refused(self):
        for value in (None, "", "d" * 63, "D" * 64, "0" * 64):
            bad = copy.deepcopy(self.report)
            bad["source"]["source_manifest_sha256"] = value
            with self.assertRaises(ValueError):
                ready_report(bad)

    def test_both_target_position_labels_required(self):
        for labels in ({"target_first": .9}, {"first": .9, "second": .9}):
            bad = copy.deepcopy(self.report)
            bad["quality"]["formats"]["raw_json"]["per_label"] = labels
            with self.assertRaises(ValueError):
                ready_report(bad)

    def test_literal_token_ids_refused(self):
        for tokens in ([True], [-1], [50368], [1.0], "1", [1] * 257):
            bad = copy.deepcopy(self.report)
            bad["runtime"]["vectors"][0]["tokens"] = tokens
            with self.assertRaises(ValueError):
                ready_report(bad)

    def test_optional_package_manifest_must_match(self):
        bad = copy.deepcopy(self.report)
        bad["package_manifest"] = {"sha256": "0" * 64, "keccak256": "0x" + "c" * 64, "bytes": 1024}
        with self.assertRaises(ValueError):
            ready_report(bad)

    def test_malformed_report_sections_raise_value_error(self):
        for section in ("source", "quality", "runtime"):
            bad = copy.deepcopy(self.report)
            bad[section] = None
            with self.assertRaises(ValueError):
                ready_report(bad)


if __name__ == "__main__":
    unittest.main()
