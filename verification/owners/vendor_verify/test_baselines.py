# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Synthetic mechanics controls. No source audit rows or neural work."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baselines import (
    decode_rows,
    evaluate_frozen,
    fit_training,
    freeze_once,
    load_coefficients,
)
from collect import packed, preference
from test_vendor import record


class BaselineTests(unittest.TestCase):
    def training(self):
        pairs = [
            preference(record(value=True)[2]),
            preference(record("995849364", "Other synthetic AS", False)[2]),
        ]
        return ("\n".join(packed(p) for p in pairs) + "\n").encode()

    def test_frozen_bytes_reused_without_fitting(self):
        raw = self.training()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.jsonl").write_bytes(raw)
            sha = freeze_once(root / "train.jsonl", root / "coefficients.json")
            value, read_sha = load_coefficients(
                root / "coefficients.json", fit_training(raw)["training_sha256"]
            )
            self.assertEqual(sha, read_sha)
            rows = decode_rows(raw)
            with patch(
                "baselines.fit_training",
                side_effect=AssertionError("refitting forbidden"),
            ):
                first = evaluate_frozen({"train": rows, "validation": rows}, value)
                self.assertEqual(
                    first, evaluate_frozen({"train": rows, "validation": rows}, value)
                )
            self.assertEqual(first["by_split"]["train"]["length"], 0.5)

    def test_validation_and_audit_never_fit_or_evaluate(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                freeze_once(
                    Path(tmp) / "validation.jsonl", Path(tmp) / "coefficients.json"
                )
        value = fit_training(self.training())
        with self.assertRaises(ValueError):
            evaluate_frozen({"audit": decode_rows(self.training())}, value)

    def test_overwrite_and_wrong_training_hash_refused(self):
        raw = self.training()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.jsonl").write_bytes(raw)
            freeze_once(root / "train.jsonl", root / "coefficients.json")
            before = (root / "coefficients.json").read_bytes()
            with self.assertRaises(FileExistsError):
                freeze_once(root / "train.jsonl", root / "coefficients.json")
            self.assertEqual(before, (root / "coefficients.json").read_bytes())
            with self.assertRaises(ValueError):
                load_coefficients(root / "coefficients.json", "0" * 64)

    def test_nonfinite_coefficient_refused(self):
        value = fit_training(self.training())
        value["idf_hex"]["bad"] = "nan"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coefficients.json"
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                load_coefficients(path, value["training_sha256"])


if __name__ == "__main__":
    unittest.main()
