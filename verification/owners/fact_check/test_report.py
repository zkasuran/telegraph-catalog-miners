# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Artificial report and literal-wire fixtures only, never audit measurements."""
import copy
import importlib.util
import sys
import unittest
from pathlib import Path

from report import require_metrics, validate_ready
from runtime_check import RELEASE_HASHES, check_response


def fixture():
    return {"schema_version": 1, "status": "REGISTRATION_READY", "family": "FACT_CHECK", "run_id": 123,
            "commit": "a" * 40, "serving_commit": "a" * 40, "package_sha256": "a" * 64, "package_keccak": "b" * 64,
            "foundation_sha256": RELEASE_HASHES["foundation_lora_fp32.onnx"], "engine_sha256": RELEASE_HASHES["foundation_engine.wasm"],
            "tokenizer_sha256": RELEASE_HASHES["tokenizer.json"],
            "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True, "independent_groups": 64,
                       "source_manifest_sha256": "a" * 64},
            "quality": {"passed": True, "audit_pairs": 64, "all_supported_formats_tested": True,
                        "formats": {"prose": {"accuracy": .9, "per_label": {"True": .9, "False": .9}, "untrained_accuracy": .5, "shortcut_accuracy": .5}}},
            "runtime": {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
                        "negative_identity_controls": 3, "vectors": [{"tokens": [50281, 1, 50282], "score_bits": 0} for _ in range(8)]}}


class ReportTests(unittest.TestCase):
    def test_report_refuses_missing_measured_quality_runtime_or_identity(self):
        value = fixture()
        validate_ready(value)
        for field in ("source", "quality", "runtime", "run_id", "package_sha256"):
            wrong = copy.deepcopy(value)
            wrong.pop(field)
            with self.subTest(field=field), self.assertRaises((ValueError, TypeError)):
                validate_ready(wrong)
        wrong = copy.deepcopy(value)
        wrong["runtime"]["vectors"][0]["score_bits"] = 0x7F800000
        with self.assertRaises(ValueError):
            validate_ready(wrong)

    def test_all_original_quality_thresholds_are_preserved(self):
        metrics = fixture()["quality"]["formats"]["prose"]
        require_metrics(metrics)
        for field, value in (("accuracy", .79), ("untrained_accuracy", .89), ("shortcut_accuracy", .66)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                require_metrics({**metrics, field: value})
        with self.assertRaises(ValueError):
            require_metrics({**metrics, "per_label": {"True": .69, "False": .99}})

    def test_literal_bare_executor_echo_is_not_a_normalized_standin(self):
        manifest = {"keccak256": "0x" + "a" * 64}
        answer = {"foundation_hash": RELEASE_HASHES["foundation_lora_fp32.onnx"], "tokenizer_hash": RELEASE_HASHES["tokenizer.json"],
                  "adapter_hash": "a" * 64, "results": [{"score": 0}]}
        self.assertEqual(check_response(answer, 1, manifest), [0])
        for value in ("0x" + "a" * 64, "A" * 64):
            with self.assertRaises(ValueError):
                check_response({**answer, "adapter_hash": value}, 1, manifest)

    def test_public_wire_rejects_prefixed_hash_and_invalid_tokens(self):
        directory = Path(__file__).resolve().parent / "wire"
        spec = importlib.util.spec_from_file_location("factcheck_wire_common", directory / "common.py")
        common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common)
        with __import__("unittest.mock", fromlist=["patch"]).patch.dict(sys.modules, {"common": common}):
            spec = importlib.util.spec_from_file_location("factcheck_wire_proxy", directory / "proxy.py")
            proxy = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(proxy)
        row = {"keccak": "a" * 64, "runnable_url": "https://tga1.zkasuran.dev/_factcheck-test/runnable"}
        request = {"batch": [{"tokens": [50281, 1, 50282]}], **proxy.identity(row)}
        self.assertEqual(proxy.validate_request(request, row)["adapter_hash"], "a" * 64)
        for wrong in ({**request, "adapter_hash": "0x" + "a" * 64}, {**request, "batch": [{"tokens": []}]},
                      {**request, "batch": [{"tokens": [True]}]}, {**request, "batch": [{"text": "not tokens"}]}):
            with self.assertRaises(ValueError):
                proxy.validate_request(wrong, row)


if __name__ == "__main__":
    unittest.main()
