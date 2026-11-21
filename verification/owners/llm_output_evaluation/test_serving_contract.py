# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Synthetic source-only report controls, never neural or node qualification.

The actual serving.start prefix executes with synthetic fetch/path/environment
doubles and stops before runtime or key preparation. No model package, private
key, reference engine or network endpoint is created or loaded.
"""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from common import DEFAULT_OWNER, FOUNDATION, immutable_url
from data import FAMILY, FORMATS, RATINGS
from serving_contract import validate_ready


PACKAGE_SHA = "a" * 64
PACKAGE_HASH = "b" * 64
SOURCE_COMMIT = "c" * 40


def synthetic_report():
    return {"schema_version": 1, "status": "REGISTRATION_READY", "family": FAMILY,
            "run_id": 11, "commit": SOURCE_COMMIT, "serving_commit": SOURCE_COMMIT,
            "package_sha256": PACKAGE_SHA, "package_keccak": PACKAGE_HASH,
            "foundation_sha256": FOUNDATION["foundation_lora_fp32.onnx"],
            "engine_sha256": FOUNDATION["foundation_engine.wasm"], "tokenizer_sha256": FOUNDATION["tokenizer.json"],
            "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True,
                       "source_manifest_sha256": "d" * 64, "independent_groups": 32},
            "quality": {"passed": True, "audit_pairs": 64, "all_supported_formats_tested": True,
                        "formats": {name: {"accuracy": 0.85, "per_label": {label: 0.8 for label in RATINGS},
                                           "untrained_accuracy": 0.75, "shortcut_accuracy": 0.60} for name in FORMATS}},
            "runtime": {"full_parser_passed": True, "reference_engine_passed": True,
                        "repeat_bits_identical": True, "negative_identity_controls": 3,
                        "vectors": [{"tokens": [50281, 22, 50282], "score_bits": 0x3f000000} for index in range(8)]}}


class ServingContractTests(unittest.TestCase):
    def check(self, report):
        return validate_ready(report, PACKAGE_SHA, PACKAGE_HASH, SOURCE_COMMIT)

    def prefix(self, report, remove_new_guard=False):
        source = Path(__file__).with_name("serving.py").read_bytes()
        start = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "start")
        prefix = []
        for node in start.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Tuple) for target in node.targets):
                break
            if (remove_new_guard and isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name) and node.value.func.id == "validate_ready"):
                continue
            prefix.append(copy.deepcopy(node))
        calls = []
        class SyntheticFile:
            def __init__(self, name):
                self.name = name
            def read_text(self):
                return json.dumps(report)
        class SyntheticRoot:
            def __truediv__(self, name):
                return SyntheticFile(name)
        prefix_url = "https://raw.githubusercontent.com/zkasuran/telegraph-catalog-miners/" + SOURCE_COMMIT
        scope = {"root": SyntheticRoot(), "json": json, "immutable_url": immutable_url,
                 "validate_ready": validate_ready,
                 "fetch": lambda url, path, sha, cap: calls.append(path.name),
                 "os": SimpleNamespace(environ={"PACKAGE_SHA256": PACKAGE_SHA, "PACKAGE_KECCAK": PACKAGE_HASH,
                                                "PACKAGE_URL": prefix_url + "/verification/owners/llm_output_evaluation/synthetic-package.safetensors",
                                                "REPORT_URL": prefix_url + "/verification/owners/llm_output_evaluation/synthetic-report.json",
                                                "REPORT_SHA256": "e" * 64, "GITHUB_SHA": SOURCE_COMMIT})}
        exec(compile(ast.Module(body=prefix, type_ignores=[]), "actual serving prefix with synthetic no-model doubles", "exec"), scope)
        return calls

    def test_valid_report_preserves_exact_vectors(self):
        report = synthetic_report()
        before = copy.deepcopy(report)
        self.assertEqual(self.check(report), report["runtime"]["vectors"])
        self.assertEqual(report, before)

    def test_actual_valid_prefix_fetches_report_before_package(self):
        self.assertEqual(self.prefix(synthetic_report()), ["registration-ready.json", "package.safetensors"])

    def test_actual_start_refuses_invalid_report_before_package(self):
        report = synthetic_report()
        report["quality"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "format audit"):
            self.prefix(report)

    def test_missing_guard_control_accepts_partial_identity_report(self):
        report = synthetic_report()
        report["quality"]["passed"] = False
        self.assertIn("package.safetensors", self.prefix(report, remove_new_guard=True))

    def test_pinned_identities_cannot_be_normalized_or_substituted(self):
        for name in ("foundation_sha256", "engine_sha256", "tokenizer_sha256"):
            report = synthetic_report()
            report[name] = "0x" + report[name]
            with self.assertRaisesRegex(ValueError, "pinned runtime identity"):
                self.check(report)

    def test_all_formats_and_rating_labels_are_required(self):
        report = synthetic_report()
        report["quality"]["formats"].pop(FORMATS[0])
        with self.assertRaises(ValueError):
            self.check(report)
        report = synthetic_report()
        report["quality"]["formats"][FORMATS[0]]["per_label"].pop(RATINGS[0])
        with self.assertRaises(ValueError):
            self.check(report)

    def test_quality_floors_baselines_and_finite_metrics(self):
        cases = (("accuracy", 0.79), ("untrained_accuracy", 0.83), ("shortcut_accuracy", 0.80),
                 ("accuracy", True), ("accuracy", float("nan")))
        for key, value in cases:
            report = synthetic_report()
            report["quality"]["formats"][FORMATS[0]][key] = value
            with self.assertRaises(ValueError):
                self.check(report)
        report = synthetic_report()
        report["quality"]["formats"][FORMATS[0]]["per_label"][RATINGS[0]] = 0.69
        with self.assertRaises(ValueError):
            self.check(report)

    def test_independent_source_boundary_required(self):
        for key, value in (("normalized_overlap", 1), ("audit_used_for_selection", True),
                           ("grants_verified", False), ("independent_groups", 31),
                           ("source_manifest_sha256", "not-a-hash")):
            report = synthetic_report()
            report["source"][key] = value
            with self.assertRaises(ValueError):
                self.check(report)

    def test_runtime_flags_and_vector_bounds(self):
        for key in ("full_parser_passed", "reference_engine_passed", "repeat_bits_identical"):
            report = synthetic_report()
            report["runtime"][key] = False
            with self.assertRaises(ValueError):
                self.check(report)
        for vectors in ([], synthetic_report()["runtime"]["vectors"][:7],
                        synthetic_report()["runtime"]["vectors"] * 13):
            report = synthetic_report()
            report["runtime"]["vectors"] = vectors
            with self.assertRaises(ValueError):
                self.check(report)
        for tokens, bits in (([True], 0), ([50368], 0), ([1], 0x7f800000), ([1], True), ([1] * 513, 0)):
            report = synthetic_report()
            report["runtime"]["vectors"][0] = {"tokens": tokens, "score_bits": bits}
            with self.assertRaises(ValueError):
                self.check(report)

    def test_top_level_run_owner_and_serving_identity(self):
        for key, value in (("package_keccak", "0x" + PACKAGE_HASH), ("serving_commit", "0" * 40),
                           ("run_id", True), ("owner", []), ("owner", "0x" + "0" * 40)):
            report = synthetic_report()
            report[key] = value
            with self.assertRaises(ValueError):
                self.check(report)


if __name__ == "__main__":
    unittest.main()
