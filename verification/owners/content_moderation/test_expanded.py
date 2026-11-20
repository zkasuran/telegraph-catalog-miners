# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Synthetic complete-source contracts, not new training data or model inference."""
import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contracts import HERE, sha
from expanded import grouped_sources, plan, select_groups, validate_plan
from remote import support


class ExpandedTests(unittest.TestCase):
    def group(self, text):
        return sha(" ".join(text.casefold().split()).encode())

    def fixture(self):
        return [{"split": split, "label": label, "text": "Synthetic " + split + " " + label + " control " + str(i)}
                for split in ("train", "validation", "test") for label in ("Safe", "Flagged") for i in range(8)]

    def test_full_scan_not_prefix_detects_late_cross_split_and_annotation_conflicts(self):
        rows = self.fixture()
        cross, conflict = rows[0]["text"], rows[1]["text"]
        rows.extend([{"split": "test", "label": "Safe", "text": cross.upper()}, {"split": "train", "label": None, "text": conflict}])
        excluded = {self.group(rows[3]["text"])}
        clean, inventory = grouped_sources(rows, excluded, self.group)
        reversed_clean, _ = grouped_sources(list(reversed(rows)), excluded, self.group)
        self.assertEqual(sorted(clean, key=lambda r: r["group_id"]), sorted(reversed_clean, key=lambda r: r["group_id"]))
        ids = {r["group_id"] for r in clean}
        self.assertFalse(ids & (excluded | {self.group(cross), self.group(conflict)}))
        self.assertEqual(inventory["raw_rows"], 50)
        selected, _ = select_groups(clean, {"train": 6, "validation": 6, "audit": 6}, {s: "synthetic-" + s for s in ("train", "validation", "audit")}, lambda row, split: True)
        self.assertEqual({s: len(r) for s, r in selected.items()}, {"train": 6, "validation": 6, "audit": 6})
        self.assertEqual(len({r["group_id"] for rows in selected.values() for r in rows}), 18)

    def test_source_only_admission_and_insufficient_groups_controls(self):
        clean, _ = grouped_sources(self.fixture(), set(), self.group)
        result, rejected = select_groups(clean, {"train": 6}, {"train": "synthetic"}, lambda row, split: not row["text"].endswith("0"))
        self.assertTrue(all(not r["text"].endswith("0") for r in result["train"]))
        self.assertGreaterEqual(rejected.get("train", 0), 0)
        with self.assertRaises(ValueError):
            select_groups(clean, {"train": 100}, {"train": "synthetic"}, lambda row, split: True)
        with self.assertRaises(ValueError):
            grouped_sources([{"split": "private", "label": "Safe", "text": "Not legitimate source"}], set(), self.group)

    def test_complete_inventory_and_frozen_recipe_controls(self):
        value = plan()
        validate_plan(value)
        for field in ("epochs", "train_groups", "validation_groups", "audit_groups", "exclusions_count", "job_minutes"):
            changed = copy.deepcopy(value)
            changed[field] = 0
            with self.assertRaises(ValueError):
                validate_plan(changed)
        changed = copy.deepcopy(value)
        changed["files"] = [r for r in changed["files"] if "train-00001" not in r["path"]]
        with self.assertRaises(ValueError):
            validate_plan(changed)

    def test_actual_packaged_source_reader_binds_prepared_bytes_and_counts(self):
        support()["core"].load_shared()
        path = HERE / "support/expanded/model_ops.py"
        spec = importlib.util.spec_from_file_location("synthetic_expanded_reader", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = support()["core"].load_shared()[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "originals").mkdir()
            (root / "evidence").mkdir()
            files = {}
            for split, count in (("train", 1536), ("validation", 256)):
                rows = []
                for i in range(count):
                    text = "Synthetic source-reader " + split + " number " + str(i)
                    rows.append(data.pair("CONTENT_MODERATION", {"text": text, "label": "Safe" if i % 2 else "Flagged", "group_id": data.group_id(text), "source_id": "synthetic", "domain": "synthetic"}))
                raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
                (root / "originals" / (split + ".jsonl")).write_bytes(raw)
                files[split] = {"sha256": sha(raw), "n": count}
            (root / "evidence/source-manifest.json").write_text(json.dumps({"sources": {"source_files": files}}))
            self.assertEqual(len(module.read_development(root)["train"]), 1536)
            target = root / "originals/train.jsonl"
            target.write_bytes(target.read_bytes() + b" ")
            with self.assertRaises(ValueError):
                module.read_development(root)

    def test_all_expanded_model_entry_points_refuse_locally_before_framework_imports(self):
        import expanded_remote
        with patch.dict(os.environ, {}, clear=True):
            for call in (expanded_remote.prepare, expanded_remote.initial, expanded_remote.select, expanded_remote.qualify):
                with self.assertRaises(RuntimeError):
                    call(None)
        for name in ("torch", "numpy", "transformers", "tokenizers", "peft"):
            self.assertNotIn(name, sys.modules)


if __name__ == "__main__":
    unittest.main()
