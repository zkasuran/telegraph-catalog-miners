# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only refusal controls, never fake local remote execution."""

import subprocess
import sys
import unittest
from pathlib import Path

import remote


class RemoteContracts(unittest.TestCase):
    def test_local_guard_precedes_imports_and_downloads(self):
        for env in (
            {},
            {"GITHUB_ACTIONS": "true"},
            {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": remote.REPO},
        ):
            with self.assertRaises(RuntimeError):
                remote.remote_only(env)
        result = subprocess.run(
            [sys.executable, str(Path(remote.__file__))],
            capture_output=True,
            text=True,
            timeout=10,
            env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("GENERAL_DEVELOPMENT_FAILED", result.stdout)
        self.assertNotIn("torch", sys.modules)

    def test_external_download_refused_before_network(self):
        with self.assertRaises(ValueError):
            remote.download(
                "https://example.invalid/weights", Path("not-created"), "0" * 64, 100
            )

    def test_development_gate_never_qualifies(self):
        metric = {"pairwise_accuracy": 0.9, "beats_length_baseline": True}
        self.assertTrue(
            remote.development_gate(metric, {"pairwise_accuracy": 0.6}, 0.6)
        )
        for accuracy, untrained, shortcut in (
            (0.79, 0.6, 0.6),
            (0.9, 0.9, 0.6),
            (0.9, 0.6, 0.85),
        ):
            self.assertFalse(
                remote.development_gate(
                    {**metric, "pairwise_accuracy": accuracy},
                    {"pairwise_accuracy": untrained},
                    shortcut,
                )
            )

    def test_source_sealed_audit_has_no_measurement_path(self):
        import ast

        tree = ast.parse(Path(remote.__file__).read_text())
        # Explicit source regression: remote model loop reads train and validation
        # only. An audit read cannot silently enter epoch selection.
        literals = [
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        ]
        self.assertNotIn("data/audit.jsonl", literals)
        self.assertNotIn("audit.jsonl", literals)


if __name__ == "__main__":
    unittest.main()
