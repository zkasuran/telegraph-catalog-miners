# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import json
import tempfile
import unittest
from pathlib import Path

from regularization import SETTINGS, load


class RegularizationChecks(unittest.TestCase):
    def test_primary_frozen_dropout_and_refuse_invalid_or_audit_selection(self):
        source = Path(__file__).with_name(SETTINGS)
        dropout, digest = load(source)
        self.assertEqual(dropout, 0.25)
        self.assertEqual(len(digest), 64)
        record = json.loads(source.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for bad in (True, -0.1, 1.0, float("nan"), "0.25"):
                path.write_text(json.dumps({**record, "lora_dropout": bad}))
                with self.assertRaises(ValueError):
                    load(path)
            path.write_text(json.dumps({**record, "selection_data": "audit"}))
            with self.assertRaises(ValueError):
                load(path)
            path.write_text(json.dumps({**record, "quality_gates_changed": True}))
            with self.assertRaises(ValueError):
                load(path)
