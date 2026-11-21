# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only scope/report controls. Synthetic metrics are never model proof."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import assets
import report
from common import FOUNDATION, KIT, sha
from contracts import validate_quality_runtime, validate_ready
from parent_scope import FAMILY, REQUEST, RESPONSE, claimed_intents, verify_scope
from test_contracts import synthetic_ready_fixture

HERE = Path(__file__).resolve().parent


class ScopeIntegration(unittest.TestCase):
    def parent(self):
        return {"intent_id": FAMILY, "intent_name": FAMILY, "class": "judgment", "prefix": True,
                "request": {"served_by": FAMILY}, "response": {"served_by": FAMILY},
                "onchain": {"served_by": FAMILY, "ready": True, "format_version": 4,
                            "request": copy.deepcopy(REQUEST), "response": copy.deepcopy(RESPONSE)}}

    def material(self, count=5):
        return {"intent": FAMILY, "corpus_version": "a" * 64, "required_contributors": 5,
                "eligible_contributors": count, "active_contributors": count,
                "material_ready": count >= 5, "child_material_borrowed": False}

    def test_scope_modes_and_legacy_general_binding(self):
        self.assertEqual(verify_scope({}, "general"), "general")
        verify_scope({"scope": "parent_only", "claimed_intents": [FAMILY]}, "parent_only")
        for value in ({}, {"scope": "parent_only", "claimed_intents": [FAMILY + "_GENERAL"]},
                      {"scope": "general", "claimed_intents": [FAMILY]}):
            with self.assertRaises(ValueError):
                verify_scope(value, "parent_only")

    def test_actual_parent_scope_import_does_not_load_neural_modules(self):
        command = [sys.executable, "-W", "error", "-c", "import assets,sys; assert assets.SCOPE == 'parent_only'; assert assets.CHILD == assets.FAMILY; assert 'torch' not in sys.modules; print('PARENT_SCOPE_IMPORT_OK')"]
        result = subprocess.run(command, cwd=HERE, capture_output=True, timeout=10,
                                env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "OWNER_SCOPE": "parent_only"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(b"PARENT_SCOPE_IMPORT_OK", result.stdout)

    def test_parent_capture_uses_real_parent_wire_not_general_override(self):
        def getter(url):
            value = self.parent() if "/engine/" in url else {"resolve": {"intent": FAMILY, "class": "judgment"}}
            return value, json.dumps(value).encode()
        with tempfile.TemporaryDirectory() as temp, patch.object(assets, "SCOPE", "parent_only"), patch.object(assets, "CHILD", FAMILY), patch.object(assets, "public_json", getter):
            envelope = assets.capture_classes(Path(temp) / "class-preflight.json")
            self.assertTrue(envelope["child"]["url"].endswith("/" + FAMILY))

    def test_parent_report_requires_explicit_material_proof(self):
        ready = synthetic_ready_fixture()
        ready.update(scope="parent_only", claimed_intents=[FAMILY])
        with self.assertRaises(ValueError):
            validate_ready(ready)
        ready["source"]["parent_material"] = self.material()
        validate_ready(ready)
        ready["source"]["parent_material"] = self.material(1)
        with self.assertRaises(ValueError):
            validate_ready(ready)

    def test_combine_preserves_material_condition_not_registration_ready(self):
        source_hash = sha((HERE / "frozen/frozen.json").read_bytes())
        ready = synthetic_ready_fixture()
        candidate = {"family": FAMILY, "scope": "parent_only", "claimed_intents": [FAMILY], "source_manifest_sha256": source_hash,
                     "audit_read": False, "run_id": 123, "commit": "a" * 40,
                     "package": {"sha256": "b" * 64, "bytes": 123}}
        reports = {}
        for name in ("json", "prose"):
            reports[name] = {"family": FAMILY, "format": name, "scope": "parent_only", "claimed_intents": [FAMILY],
                             "passed": True, "run_id": 123, "commit": "a" * 40, "package_sha256": "b" * 64,
                             "source_manifest_sha256": source_hash, "foundation": FOUNDATION, "kit_commit": KIT,
                             "quality_engine": "pinned reference WASM, no native Torch audit", "package_keccak": "c" * 64,
                             "runtime": ready["runtime"], "executor_sha256": "d" * 64,
                             "independent_groups": 39, "audit_pairs": 384, "metrics": ready["quality"]["formats"][name],
                             "runtime_versions": {}, "class_snapshot_sha256": "e" * 64}
        status = {"intent_id": FAMILY, "eligible_submitters": 1, "min_submitters": 3,
                  "active_snapshot": {"corpus_version": "f" * 64, "submitters": 1, "rows": 150, "qualification": 78, "ranking": 72}}
        with tempfile.TemporaryDirectory() as temp, patch.object(report, "SCOPE", "parent_only"), patch.object(report, "public_json", lambda _: (status, json.dumps(status).encode())):
            output = Path(temp) / "registration-ready.json"
            with self.assertRaises(ValueError):
                report.combine(candidate, reports, output)
            condition = json.loads(output.with_name("material-condition.json").read_text())
            self.assertEqual(condition["status"], "WAITING_PARENT_MATERIAL")
            self.assertFalse(condition["registration_ready"])
            self.assertFalse(output.exists())

    def test_training_and_parser_scope_calls_are_explicit(self):
        for name in ("remote.py", "runtime.py", "report.py", "serving.py"):
            tree = ast.parse((HERE / name).read_text())
            self.assertTrue(any(isinstance(node, ast.Name) and node.id == "SCOPE" for node in ast.walk(tree)), name)
        self.assertIn('"--scope", SCOPE', (HERE / "runtime.py").read_text())
        self.assertIn('training_record(len(validation), validation_accuracy)', (HERE / "remote.py").read_text())

    def test_material_wait_never_masks_failed_quality_or_runtime(self):
        for section, key, value in (("quality", "audit_pairs", 63), ("runtime", "reference_engine_passed", False)):
            ready = synthetic_ready_fixture()
            ready[section][key] = value
            with self.assertRaises(ValueError):
                validate_quality_runtime(ready)
        tree = ast.parse((HERE / "report.py").read_text())
        combine = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "combine")
        calls = [node for node in ast.walk(combine) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertLess(min(node.lineno for node in calls if node.func.id == "validate_quality_runtime"),
                        min(node.lineno for node in calls if node.func.id == "material_contract"))

    def test_workflow_scope_is_bound_and_failed_conditions_are_retained(self):
        workflow = json.loads((HERE.parents[2] / ".github/workflows/evaluator-owner-event_outcome_resolution.yml").read_text())
        inputs = workflow["on"]["workflow_dispatch"]["inputs"]
        self.assertLessEqual(len(inputs), 10)
        self.assertEqual(inputs["mode"]["options"], ["verify", "verify_parent", "serve", "serve_parent"])
        self.assertEqual(workflow["env"]["OWNER_SCOPE"], "${{ (inputs.mode == 'verify_parent' || inputs.mode == 'serve_parent') && 'parent_only' || 'general' }}")
        upload = workflow["jobs"]["report"]["steps"][-1]
        self.assertEqual(upload["if"], "always()")
        self.assertEqual(upload["with"]["if-no-files-found"], "error")


if __name__ == "__main__":
    unittest.main()
