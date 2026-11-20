# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only tests of recovery boundaries. No tensor/model imports."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest

from recover import ORIGINAL_COMMIT, ORIGINAL_RUN, PACKAGE_SHA, frozen_answer_control

HERE = Path(__file__).resolve().parent


class RecoveryContracts(unittest.TestCase):
    def test_exact_original_candidate_and_no_retraining(self):
        spec = json.loads((HERE / "recovery-spec.json").read_text())
        self.assertEqual(spec["package_sha256"], PACKAGE_SHA)
        self.assertEqual(spec["original_run_id"], ORIGINAL_RUN)
        self.assertEqual(spec["original_commit"], ORIGINAL_COMMIT)
        self.assertEqual(spec["expected_max_tokens"], 512)
        self.assertFalse(spec["training_allowed"])
        self.assertFalse(spec["registration_ready"])

    def test_source_has_no_training_or_selection_calls(self):
        source = ast.parse((HERE / "recover.py").read_text())
        called = {n.func.attr for n in ast.walk(source) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse({"backward", "step", "train"} & called)
        self.assertIn("read_package", (HERE / "recover.py").read_text())
        self.assertIn("all_tensor_bytes_identical", (HERE / "recover.py").read_text())

    def test_remote_refusal_before_model_or_files(self):
        root = HERE / "must-not-create"
        result = subprocess.run([sys.executable, str(HERE / "recover.py"), "--root", str(root)],
                                env={"PATH": "/usr/bin:/bin", "GITHUB_ACTIONS": "false", "PYTHONDONTWRITEBYTECODE": "1"},
                                capture_output=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"bounded owned standard public CPU", result.stderr)
        self.assertFalse(root.exists())

    def test_frozen_baseline_rejects_cross_data_binding_before_numpy(self):
        class Bytes:
            def read_text(self):
                return '{"fitted_on":"validation", "train_sha256":"wrong"}'
        with self.assertRaisesRegex(ValueError, "binding"):
            frozen_answer_control(None, [], Bytes(), "expected")

    def test_actual_artifact_data_is_pinned(self):
        spec = json.loads((HERE / "recovery-spec.json").read_text())
        data = json.loads((HERE / "original-data-report.json").read_text())
        self.assertEqual(data["files"]["train"]["sha256"], spec["train_sha256"])
        self.assertEqual(data["files"]["validation"]["sha256"], spec["validation_sha256"])
        self.assertEqual(data["files"]["train"]["pairs"], 1184)
        self.assertEqual(data["files"]["validation"]["pairs"], 1216)
        self.assertFalse(data["audit_available"])


if __name__ == "__main__":
    unittest.main()
