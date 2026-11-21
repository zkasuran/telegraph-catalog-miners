# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Class-source preflight controls without local neural operations."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import assets


class Preflight(unittest.TestCase):
    def fixture(self, child_class="judgment", served_by=assets.FAMILY):
        return {"intent_id": assets.CHILD, "intent_name": assets.CHILD, "class": child_class,
                "evaluation": {"kind": child_class}, "request": {"served_by": served_by},
                "response": {"served_by": served_by}, "onchain": {"served_by": served_by, "ready": True}}

    def getter(self, child):
        def get(url):
            value = child if "/engine/" in url else {"resolve": {"intent": assets.FAMILY, "class": "judgment"}}
            return value, json.dumps(value).encode()
        return get

    def test_valid_class_snapshot_primary_path(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(assets, "public_json", self.getter(self.fixture())):
            out = Path(temp) / "class-preflight.json"
            assets.capture_classes(out)
            self.assertTrue(out.is_file())
            self.assertFalse(out.with_suffix(".failure.json").exists())

    def test_live_hybrid_self_routing_refused_and_diagnosed(self):
        # Invented redaction sentinel, not a row from any evaluation pool.
        child = {**self.fixture("hybrid", assets.CHILD), "extra_rows": [{"must_not_export": "sentinel"}]}
        with tempfile.TemporaryDirectory() as temp, patch.object(assets, "public_json", self.getter(child)):
            out = Path(temp) / "class-preflight.json"
            with self.assertRaises(ValueError):
                assets.capture_classes(out)
            report = json.loads(out.with_suffix(".failure.json").read_text())
            self.assertEqual(report["status"], "SOURCE_CLASS_GATE_FAILED")
            self.assertEqual(report["observed_child_class"], "hybrid")
            self.assertEqual(report["stage"], "preflight")
            self.assertNotIn("must_not_export", json.dumps(report))
            self.assertFalse(out.exists())

    def test_class_capture_precedes_training_inputs_and_assets(self):
        tree = ast.parse(Path(__file__).with_name("remote.py").read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        preflight = min(node.lineno for node in calls if node.func.id == "capture_classes")
        self.assertLess(preflight, min(node.lineno for node in calls if node.func.id in ("read_split", "prepare")))


if __name__ == "__main__":
    unittest.main()
