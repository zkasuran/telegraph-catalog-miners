# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote guard, complete quality coverage and unchanged helper controls."""
import copy
import math
import os
import struct
import subprocess
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from remote_runtime import FOUNDATION, metrics, quality
from reuse_helpers import package_source, runtime_functions, source

HERE = Path(__file__).resolve().parent


class RemoteTests(unittest.TestCase):
    def test_shared_ast_source_is_hash_checked_without_execution(self):
        from remote_train import trusted_prepare_ast
        tree = trusted_prepare_ast((HERE.parent / "training/run.py").read_bytes())
        self.assertEqual([node.name for node in tree.body], ["prepare"])
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            trusted_prepare_ast(b"raise RuntimeError('untrusted source must not run')")

    def test_only_generated_family_constant_changes_in_controller_parser(self):
        self.assertEqual(package_source().replace('const familyID = "LANGUAGE_TRANSLATION"', 'const familyID = "SENTIMENT_ANALYSIS"', 1), source("packagecheck.go.txt"))
        self.assertIn("adapterbuild.CheckLoRA", package_source())

    def test_controller_bit_and_identity_controls(self):
        env = runtime_functions({"struct": struct, "math": math, "contextmanager": contextmanager,
                                 "RELEASE_HASHES": FOUNDATION})
        self.assertEqual(env["decode_bits"](0x3f800000), 1.0)
        for value in (True, -1, 0x7fc00000, 0x7f800000):
            with self.assertRaises(ValueError):
                env["decode_bits"](value)
        manifest = {"keccak256": "a" * 64}
        response = {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"], "tokenizer_hash": FOUNDATION["tokenizer.json"],
                    "adapter_hash": "a" * 64, "results": [{"score": 0x3f800000}]}
        self.assertEqual(env["check_response"](response, 1, manifest), [0x3f800000])
        for name in ("foundation_hash", "tokenizer_hash", "adapter_hash"):
            bad = {**response, name: "0" * 64}
            with self.assertRaises(ValueError):
                env["check_response"](bad, 1, manifest)

    def test_model_main_refuses_local_execution_before_model_import(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("GITHUB_") and k != "RUNNER_ENVIRONMENT"}
        result = subprocess.run([sys.executable, str(HERE / "remote_train.py"), "--root", str(HERE / "out/never-created-model")],
                                env=env, capture_output=True, timeout=10, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"only in an explicitly enabled", result.stderr)
        self.assertFalse((HERE / "out/never-created-model").exists())
        self.assertFalse({"torch", "transformers", "peft", "onnxruntime"}.intersection(sys.modules))

    def test_quality_requires_all_formats_and_unchanged_thresholds(self):
        metric = {"pairwise_accuracy": 0.9, "subsets": {k: 0.9 for k in ("eng-fra", "fra-eng", "chosen_longer", "chosen_shorter")}}
        trained = {k: copy.deepcopy(metric) for k in ("raw", "json", "prose")}
        initial = {k: {"pairwise_accuracy": 0.5} for k in trained}
        self.assertTrue(quality(trained, initial, 0.53)["passed"])
        for bad_value in (0.79, float("nan")):
            bad = copy.deepcopy(trained)
            bad["json"]["pairwise_accuracy"] = bad_value
            self.assertFalse(quality(bad, initial, 0.53)["passed"])
        bad = copy.deepcopy(trained)
        bad["prose"]["subsets"]["fra-eng"] = 0.69
        self.assertFalse(quality(bad, initial, 0.53)["passed"])
        self.assertFalse(quality({"raw": metric}, {"raw": {"pairwise_accuracy": 0.5}}, 0.53)["passed"])
        self.assertFalse(quality(trained, initial, float("nan"))["passed"])

    def test_partial_final_coverage_refused(self):
        with self.assertRaises(ValueError):
            metrics([], [])

    def test_failed_guard_even_with_partial_github_environment(self):
        from checkpoint_terms import remote_guard
        with (
            patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "zkasuran/telegraph-catalog-miners", "RUNNER_ENVIRONMENT": "self-hosted"}),
            self.assertRaises(RuntimeError),
        ):
            remote_guard()


if __name__ == "__main__":
    unittest.main()
