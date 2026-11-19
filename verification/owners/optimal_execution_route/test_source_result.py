# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Constructed file-staging controls, not remote audit or model data."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from release_gate import workflow_check
from stage_source_result import FILES, MAX_FILE_BYTES, stage
from sources import HERE
from source_import_guard import SourceOnly


def fixture(root):
    for name in FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("constructed stager fixture\n")
    source = {"status": "SOURCE_GATE_PASSED", "registration_ready": False, "kind": "constructed stager regression"}
    raw = json.dumps(source).encode()
    (root / ".gate-work/source-report.json").write_bytes(raw)
    remote = {"status": "SOURCE_VERIFIED_NOT_REGISTRATION_READY", "registration_ready": False,
              "source_report_sha256": hashlib.sha256(raw).hexdigest(), "kind": "constructed stager regression"}
    (root / ".gate-work/remote-source-report.json").write_text(json.dumps(remote))


class SourceResultTests(unittest.TestCase):
    def test_source_only_guard_refuses_neural_import_names_without_loading(self):
        guard = SourceOnly()
        for name in ("torch", "tensorflow", "onnxruntime", "transformers", "peft", "torch.nn"):
            with self.assertRaises(RuntimeError):
                guard.find_spec(name)
        self.assertIsNone(guard.find_spec("json"))

    def test_stages_reports_and_not_cache_binary_or_extra_files(self):
        with tempfile.TemporaryDirectory(prefix="route-result-") as directory:
            root = Path(directory)
            fixture(root)
            (root / ".gate-work/private-extra.txt").write_text("not a permitted source report")
            (root / ".gate-work/compiled-binary").write_bytes(b"not a permitted binary")
            report = stage(root)
            self.assertTrue(report["complete"])
            self.assertEqual({r["path"] for r in report["files"]}, set(FILES.values()))
            self.assertFalse((root / "result/private-extra.txt").exists())
            self.assertFalse((root / "result/compiled-binary").exists())

    def test_missing_report_and_changed_binding_fail(self):
        with tempfile.TemporaryDirectory(prefix="route-result-") as directory:
            root = Path(directory)
            fixture(root)
            remote = root / ".gate-work/remote-source-report.json"
            remote.unlink()
            partial = stage(root, allow_partial=True)
            self.assertFalse(partial["complete"])
            with self.assertRaises(ValueError):
                stage(root)
            fixture(root)
            (root / ".gate-work/source-report.json").write_text("{}")
            with self.assertRaises(ValueError):
                stage(root)

    def test_unknown_result_payload_cannot_be_uploaded(self):
        with tempfile.TemporaryDirectory(prefix="route-result-") as directory:
            root = Path(directory)
            fixture(root)
            (root / "result").mkdir()
            (root / "result/private-extra.txt").write_text("not an allowlisted report")
            with self.assertRaises(ValueError):
                stage(root)

    def test_symlink_and_size_limits_fail(self):
        with tempfile.TemporaryDirectory(prefix="route-result-") as directory:
            root = Path(directory)
            fixture(root)
            path = root / ".gate-work/lint.log"
            path.unlink()
            path.symlink_to(root / ".gate-work/tests.log")
            with self.assertRaises(ValueError):
                stage(root)
            path.unlink()
            path.write_bytes(b"x" * (MAX_FILE_BYTES + 1))
            with self.assertRaises(ValueError):
                stage(root)

    def test_workflow_cannot_reintroduce_hidden_work_upload(self):
        path = HERE.parents[2] / ".github/workflows/evaluator-owner-optimal_execution_route.yml"
        original = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        changed = copy.deepcopy(original)
        uploader = next(s for s in changed["jobs"]["source"]["steps"] if s.get("uses", "").startswith("actions/upload-artifact@"))
        uploader["with"]["path"] = "verification/owners/optimal_execution_route/.gate-work/"
        with tempfile.TemporaryDirectory(prefix="route-result-workflow-") as directory:
            bad = Path(directory) / "workflow.yml"
            bad.write_text(yaml.safe_dump(changed))
            with self.assertRaises(ValueError):
                workflow_check(bad)


if __name__ == "__main__":
    unittest.main()
