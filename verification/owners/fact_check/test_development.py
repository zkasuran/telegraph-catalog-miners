# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""No-model source smoke and meaningful audit-access failure controls."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data import HERE
from development_policy import RECIPE, prepare_development, quality, select


class DevelopmentTests(unittest.TestCase):
    def test_real_primary_path_never_opens_audit(self):
        original = Path.read_bytes
        def audited_read(path):
            if path.name == "audit.jsonl":
                raise AssertionError("development opened the exposed audit")
            return original(path)
        with tempfile.TemporaryDirectory(dir=HERE) as temporary, patch.object(Path, "read_bytes", audited_read):
            rows, pins = prepare_development(Path(temporary))
            self.assertEqual({name: len(values) for name, values in rows.items()}, {"train": 128, "validation": 32})
            self.assertEqual(set(pins), {"train", "validation"})
            self.assertFalse((Path(temporary) / "audit.jsonl").exists())

    def test_audit_material_in_development_sandbox_is_refused(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            (root / "audit.jsonl").write_bytes(b"artificial sentinel, not real audit rows")
            with self.assertRaisesRegex(ValueError, "forbidden audit"):
                prepare_development(root)

    def test_selection_is_fixed_and_loss_only(self):
        self.assertEqual(RECIPE["epochs"], 24)
        best, _ = select(None, 1, {"val_loss": .4, "audit_accuracy": 1})
        best, improved = select(best, 2, {"val_loss": .3, "audit_accuracy": 0})
        self.assertTrue(improved)
        self.assertEqual(best["epoch"], 2)
        self.assertFalse(select(best, 3, {"val_loss": .3})[1])
        with self.assertRaises(ValueError):
            select(best, 25, {"val_loss": .1})

    def test_true_preference_and_missing_labels_cannot_pass(self):
        controls = {key: .5 for key in ("length", "token_length", "overlap", "answer_only", "surface")}
        good = {"pairwise_accuracy": .9, "accuracy_by_label": {"True": .9, "False": .9}}
        self.assertTrue(quality(good, {"pairwise_accuracy": .5}, controls)["passed"])
        self.assertFalse(quality({"pairwise_accuracy": .5, "accuracy_by_label": {"True": 1., "False": 0.}},
                                 {"pairwise_accuracy": .5}, controls)["passed"])
        self.assertFalse(quality({**good, "accuracy_by_label": {"True": .9}}, {"pairwise_accuracy": .5}, controls)["passed"])
        self.assertFalse(quality(good, {"pairwise_accuracy": .89}, controls)["passed"])


if __name__ == "__main__":
    unittest.main()
