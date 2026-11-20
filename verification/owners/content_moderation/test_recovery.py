# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Real source/ZIP/process controls only. No local model execution."""
import copy
import io
import json
import os
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from contracts import HERE, sha
from recover import archive_members, measure, prepare, validate_plan
from remote import execute_phase


class RecoveryTests(unittest.TestCase):
    def fixture(self, name="cm-training/evidence/control.json"):
        value = b'{"scope":"Synthetic ZIP source control, not measured quality"}\n'
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr(name, value)
        raw = stream.getvalue()
        plan = {"archive_sha256": sha(raw), "archive_bytes": len(raw),
                "files": [{"member": name, "name": "control.json", "sha256": sha(value), "bytes": len(value)}]}
        return raw, plan, value

    def test_primary_zip_roundtrip_and_byte_tamper_refusal(self):
        raw, plan, value = self.fixture()
        self.assertEqual(archive_members(raw, plan), {"control.json": value})
        for changed in (raw[:-1], b"x" + raw[1:]):
            with self.assertRaises(ValueError):
                archive_members(changed, plan)
        bad = copy.deepcopy(plan)
        bad["files"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            archive_members(raw, bad)

    def test_real_zip_traversal_absolute_and_backslash_refuse(self):
        for name in ("../control.json", "/control.json", "cm-training\\control.json"):
            raw, plan, _ = self.fixture(name)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                archive_members(raw, plan)

    def test_frozen_origin_audit_and_budget_plan_controls(self):
        original = json.loads((HERE / "recovery.json").read_bytes())
        validate_plan(original)
        for key, value in (("origin_run_id", 1), ("origin_commit", "0" * 40), ("artifact_id", 1),
                           ("archive_sha256", "0" * 64), ("audit_sha256", "0" * 64),
                           ("phase_cap_seconds", 0), ("url", "https://example.com/archive.zip"),
                           ("trained_score_streams", {"json": 512})):
            changed = copy.deepcopy(original)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_plan(changed)
        changed = copy.deepcopy(original)
        changed["files"][1] = changed["files"][0]
        with self.assertRaises(ValueError):
            validate_plan(changed)

    def test_real_non_neural_phase_timeout_preserves_log_and_kills_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evidence").mkdir()
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                execute_phase(root, "control", [sys.executable, "-c", "import time; print('primary-path-started', flush=True); time.sleep(60)"], 0.25, time.monotonic() + 150)
            self.assertLess(time.monotonic() - started, 5)
            self.assertIn("primary-path-started", (root / "evidence/control.log").read_text())
            self.assertEqual(json.loads((root / "evidence/phase.json").read_bytes())["state"], "running")

    def test_recovery_guard_refuses_before_framework_imports(self):
        with patch.dict(os.environ, {}, clear=True):
            for call in (prepare, measure):
                with self.assertRaises(RuntimeError):
                    call(None)
        for name in ("torch", "transformers", "tokenizers", "peft", "numpy"):
            self.assertNotIn(name, sys.modules)


if __name__ == "__main__":
    unittest.main()
