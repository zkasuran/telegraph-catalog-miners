# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only v3 controls, using constructed fixtures and no neural imports."""
import copy
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import v3_data as data
import v3_remote as remote
from contracts import file_sha, package_parts, strict_json


class V3Tests(unittest.TestCase):
    def fixture(self):
        rows = []
        for split in ("train", "validation", "audit"):
            for label in data.LABELS[:3]:
                for index in range(4 if label == "Neutral" else 7):
                    text = f"Constructed test fixture {split} {label} case {index}"
                    uid = data.group_id(text)
                    rows.append({"text": text, "source_id": uid, "author": "author-" + uid,
                                 "thread": "thread-" + uid, "group_id": uid, "timestamp": index,
                                 "label": label, "split": split})
        return rows

    def empty_quarantine(self):
        return {key: [] for key in ("sources", "entities", "threads", "source_groups", "constructed_groups")}

    def selected(self):
        with patch.dict(data.COUNTS, {"train": 3, "validation": 3, "audit": 3}, clear=True):
            return data.select_groups(self.fixture(), set())[0]

    def metrics(self, accuracy=.9, label=.9, loss=.2):
        return {"pairwise_accuracy": accuracy, "val_loss": loss,
                "formats": {name: {"accuracy": accuracy, "per_label": {value: label for value in data.LABELS}}
                            for name in ("json", "prose", "bare_label")}}

    def test_all_labels_have_composition_controls_and_disjoint_originals(self):
        groups = self.selected()
        with patch.dict(data.COUNTS, {"train": 3, "validation": 3, "audit": 3}, clear=True):
            data.check_groups(groups, self.empty_quarantine())
        for rows in groups.values():
            self.assertEqual(len(rows), 12)
            self.assertEqual({row["label"] for row in rows if row["composition"] == 2}, set(data.LABELS))

    def test_quarantined_audit_source_entity_and_thread_are_refused(self):
        for key in ("sources", "entities", "threads", "source_groups"):
            groups = self.selected()
            quarantine = self.empty_quarantine()
            quarantine[key] = [groups["train"][0][key][0]]
            with patch.dict(data.COUNTS, {"train": 3, "validation": 3, "audit": 3}, clear=True):
                with self.assertRaises(ValueError):
                    data.check_groups(groups, quarantine)

    def test_duplicates_and_cross_split_originals_are_refused(self):
        groups = self.selected()
        groups["audit"][0]["sources"] = groups["train"][0]["sources"]
        with patch.dict(data.COUNTS, {"train": 3, "validation": 3, "audit": 3}, clear=True):
            with self.assertRaises(ValueError):
                data.check_groups(groups, self.empty_quarantine())

    def test_every_format_contains_all_label_contrasts_and_short_evidence(self):
        groups = self.selected()
        for style in range(3):
            pairs = data.pairs(groups["audit"], style)
            self.assertEqual(len(pairs), 36)
            self.assertTrue(all(row["label"] != row["wrong_label"] and row["chosen"] != row["rejected"] for row in pairs))
            if style == 0:
                self.assertTrue(all(-1 <= strict_json(row["chosen"])["score"] <= 1 for row in pairs))
        with self.assertRaises(ValueError):
            data.pairs(groups["audit"], 3)

    def test_selection_prioritizes_worst_label_over_favorable_average_or_loss(self):
        uniform = self.metrics(.86, .82, .3)
        misleading = self.metrics(.94, .9, .1)
        misleading["formats"]["bare_label"]["per_label"]["Mixed"] = .34
        self.assertGreater(remote.selection_key(uniform), remote.selection_key(misleading))

    def test_validation_cannot_omit_a_format_label_or_finite_metric(self):
        for kind in ("format", "label", "nonfinite"):
            metrics = copy.deepcopy(self.metrics())
            if kind == "format":
                metrics["formats"].pop("bare_label")
            elif kind == "label":
                metrics["formats"]["json"]["per_label"].pop("Mixed")
            else:
                metrics["val_loss"] = float("nan")
            with self.assertRaises(ValueError):
                remote.selection_key(metrics)

    def test_new_neural_entrypoints_refuse_local_execution_before_imports(self):
        with patch.dict(os.environ, {}, clear=True):
            for function in (remote.train, remote.audit):
                with self.assertRaises(RuntimeError):
                    function(Path("source-only-fixture"))
            with self.assertRaises(RuntimeError):
                remote.evaluate(None, None, None, None)
            with self.assertRaises(RuntimeError):
                remote.checkpoint(None, None, None, None, None, None, None)

    def test_frozen_configuration_quarantine_and_failed_checkpoint_grants(self):
        cfg = remote.config()
        self.assertEqual(cfg["baseline_sha256"], file_sha(data.BASELINE))
        self.assertEqual(cfg["quarantine_sha256"], file_sha(data.QUARANTINE))
        self.assertEqual(cfg["warm_package_sha256"], file_sha(remote.WARM))
        self.assertEqual(len(strict_json(data.QUARANTINE.read_bytes())["sources"]), 160)
        self.assertEqual(strict_json(data.BASELINE.read_bytes())["train_sha256"], cfg["train_sha256"])
        meta = package_parts(remote.WARM.read_bytes())[0]["__metadata__"]
        self.assertIn("LicenseRef-zkasuran-SAND-1.0", meta["x-licence"])
        self.assertIn("Apache License", meta["x-goemotions-apache-2.0"])
        self.assertEqual(meta["intent"], "SENTIMENT_ANALYSIS")


if __name__ == "__main__":
    unittest.main()
