# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Local source-only tests. No neural imports, fitting or engine execution."""
import copy
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import contracts

HERE = Path(__file__).resolve().parent


class LearnerTests(unittest.TestCase):
    def test_frozen_grouped_data(self):
        data = contracts.read_development(HERE / "data")
        self.assertEqual(len(data["train"]), 256)
        self.assertEqual(len(data["validation"]), 64)
        for name in data:
            self.assertEqual({row["format"] for row in data[name]}, contracts.FORMATS)
            self.assertEqual({row["label"] for row in data[name]}, contracts.LABELS)
            self.assertEqual(contracts.margin_accuracy([contracts.basic_margins(row)["length"] for row in data[name]]), 0.5)
            self.assertEqual(contracts.margin_accuracy([contracts.basic_margins(row)["overlap"] for row in data[name]]), 0.5)

    def test_remote_guard_and_actual_primary_entry_refusal(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "bounded owned"):
                contracts.remote_only()
        process = subprocess.run([sys.executable, "-W", "error", str(HERE / "remote.py"), "--root", "unused"],
                                 env={"PATH": os.environ["PATH"], "GITHUB_ACTIONS": "false", "PYTHONDONTWRITEBYTECODE": "1"},
                                 text=True, capture_output=True, timeout=10)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("bounded owned public CPU workflow", process.stderr)
        self.assertNotIn("torch", process.stderr)
        self.assertFalse((HERE / "unused").exists())

    def test_literal_primary_render(self):
        self.assertEqual(contracts.render(" Q\r\n", " A\n"), "### Task:\n Q\r\n\n\n### Answer:\n A\n")
        self.assertNotEqual(contracts.render(" Q ", " A "), contracts.render("Q", "A"))

    def metrics(self):
        return {fmt: {"accuracy": 0.9, "per_label": {label: 0.9 for label in contracts.LABELS}} for fmt in contracts.FORMATS}

    def test_unchanged_per_format_quality_gates(self):
        good = self.metrics()
        baseline = {fmt: {"accuracy": 0.7} for fmt in contracts.FORMATS}
        shortcut = {fmt: 0.5 for fmt in contracts.FORMATS}
        self.assertTrue(contracts.validation_gate(good, baseline, shortcut)["passed"])
        for value in (0.79, float("nan"), float("inf")):
            changed = copy.deepcopy(good)
            changed["json"]["accuracy"] = value
            self.assertFalse(contracts.validation_gate(changed, baseline, shortcut)["passed"])
        changed = copy.deepcopy(good)
        changed["prose"]["per_label"][next(iter(contracts.LABELS))] = 0.69
        self.assertFalse(contracts.validation_gate(changed, baseline, shortcut)["passed"])
        self.assertFalse(contracts.validation_gate(good, {fmt: {"accuracy": 0.86} for fmt in contracts.FORMATS}, shortcut)["passed"])
        self.assertFalse(contracts.validation_gate(good, baseline, {fmt: 0.81 for fmt in contracts.FORMATS})["passed"])
        self.assertFalse(contracts.validation_gate({"json": good["json"]}, baseline, shortcut)["passed"])

    def test_train_and_validation_entity_keys_remain_disjoint(self):
        report = json.loads((HERE / "data/development-report.json").read_text())
        train, validation = report["entities"]["train"], report["entities"]["validation"]
        self.assertFalse(set(train["asns"]) & set(validation["asns"]))
        self.assertFalse(set(train["networks"]) & set(validation["networks"]))
        self.assertEqual(report["audit_pairs"], 0)
        self.assertFalse(report["audit_used_for_selection"])


if __name__ == "__main__":
    unittest.main()
