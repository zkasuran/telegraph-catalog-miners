# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only retraining contracts. All small fixtures are constructed test data."""
import copy
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import retrain
import retrain_data as data
from contracts import HERE, file_sha, strict_json, validate_ready
from test_pipeline import ready


class RetrainTests(unittest.TestCase):
    def test_all_source_files_have_real_hashes_and_original_grants(self):
        sources = strict_json(data.SOURCE_FILE.read_bytes())
        self.assertEqual(sources["stage"], "verified")
        self.assertEqual(len(sources["files"]), 6)
        for item in sources["files"].values():
            self.assertRegex(item["sha256"], r"^[a-f0-9]{64}$")
            self.assertGreater(item["bytes"], 0)

    def test_configuration_and_frozen_baseline_bind_only_training(self):
        cfg = retrain.config()
        for key in ("source_manifest_sha256", "sources_sha256", "baseline_sha256", "train_sha256",
                    "validation_sha256", "audit_sha256", "warm_package_sha256"):
            self.assertRegex(cfg[key], r"^[a-f0-9]{64}$")
        self.assertEqual(cfg["baseline_sha256"], file_sha(HERE / "retrain-baseline.json"))
        self.assertEqual(cfg["sources_sha256"], file_sha(data.SOURCE_FILE))
        self.assertEqual(strict_json(data.BASELINE_FILE.read_bytes())["train_sha256"], cfg["train_sha256"])
        self.assertEqual(cfg["hyperparameters"]["epochs"], 2)
        self.assertLessEqual(cfg["hyperparameters"]["max_length"], 512)

    def fixture(self):
        rows = []
        for split in ("train", "validation", "audit"):
            for label in data.LABELS[:3]:
                for index in range(1 if label == "Neutral" else 2):
                    text = f"Synthetic contract fixture {split} {label} {index}"
                    uid = data.group_id(text)
                    rows.append({"text": text, "source_id": uid, "group_id": uid,
                                 "author": "author-" + uid, "thread": "thread-" + uid,
                                 "timestamp": index, "label": label, "split": split})
        return rows

    def test_grouped_primary_path_has_four_labels_and_distinct_originals(self):
        with patch.dict(data.COUNTS, {"train": 1, "validation": 1, "audit": 1}, clear=True):
            groups, _ = data.grouped_select(self.fixture(), set())
            data.check_groups(groups)
            self.assertTrue(all(len(rows) == 4 for rows in groups.values()))
            for rows in groups.values():
                mixed = next(r for r in rows if r["label"] == "Mixed")
                self.assertEqual(len(mixed["sources"]), 2)
                self.assertIn("constructed", mixed["construction"])

    def test_cross_split_author_thread_or_text_cannot_enter_selection(self):
        for key in ("author", "thread", "group_id"):
            rows = self.fixture()
            rows[5][key] = rows[0][key]
            with patch.dict(data.COUNTS, {"train": 1, "validation": 1, "audit": 1}, clear=True):
                with self.assertRaises(ValueError):
                    data.grouped_select(rows, set())

    def test_prior_source_overlap_is_not_recycled_into_new_audit(self):
        rows = self.fixture()
        with patch.dict(data.COUNTS, {"train": 1, "validation": 1, "audit": 1}, clear=True):
            with self.assertRaises(ValueError):
                data.grouped_select(rows, {r["group_id"] for r in rows})

    def test_every_label_has_all_three_wrong_label_contrasts(self):
        with patch.dict(data.COUNTS, {"train": 1, "validation": 1, "audit": 1}, clear=True):
            groups, _ = data.grouped_select(self.fixture(), set())
            for style in range(3):
                pairs = data.make_pairs(groups["audit"], style)
                self.assertEqual(len(pairs), 12)
                self.assertEqual({r["label"] for r in pairs}, set(data.LABELS))
                self.assertTrue(all(r["chosen"] != r["rejected"] for r in pairs))
            with self.assertRaises(ValueError):
                data.make_pairs(groups["audit"], 3)

    def test_expanded_report_cannot_omit_neutral_mixed_or_lower_their_gate(self):
        report = ready()
        report["supported_labels"] = list(data.LABELS)
        for metrics in report["quality"]["formats"].values():
            metrics["per_label"].update(Neutral=.9, Mixed=.9)
        validate_ready(report)
        for label in ("Neutral", "Mixed"):
            for omit in (False, True):
                broken = copy.deepcopy(report)
                if omit:
                    broken["quality"]["formats"]["json"]["per_label"].pop(label)
                else:
                    broken["quality"]["formats"]["json"]["per_label"][label] = .69
                with self.assertRaises(ValueError):
                    validate_ready(broken)

    def test_invalid_or_repeated_claimed_label_inventory_is_refused(self):
        for labels in (["Positive", "Positive"], ["Other"], [["Positive"]]):
            with self.assertRaises(ValueError):
                validate_ready({**ready(), "supported_labels": labels})

    def test_all_new_neural_entrypoints_fail_before_any_local_model_operation(self):
        with patch.dict(os.environ, {}, clear=True):
            for function in (retrain.train, retrain.audit, retrain.base_assets):
                with self.assertRaises(RuntimeError):
                    function(Path("source-only-fixture"))
            with self.assertRaises(RuntimeError):
                retrain.warm_start(None, None)
            with self.assertRaises(RuntimeError):
                retrain.save_checkpoint(None, None, None, None, None, None, None)


if __name__ == "__main__":
    unittest.main()
