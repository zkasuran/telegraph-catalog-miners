# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only actual selected-state resume controls. No neural imports."""
import copy
import json
import unittest

from common import HERE, sha
from checkpoint_contract import check_checkpoint
from resume_contract import CHECKPOINT, READOUT, COMMIT, resume_policy, source_resume


class Rank16ResumeTests(unittest.TestCase):
    def setUp(self):
        self.recipe = json.loads((HERE / "recipe.json").read_text())

    def test_real_rank16_adapter_and_paired_readout_are_preserved(self):
        policy = resume_policy(self.recipe)
        raw = (HERE / self.recipe["checkpoint_file"]).read_bytes()
        report = source_resume(raw, self.recipe)
        self.assertEqual(report["sha256"], CHECKPOINT)
        self.assertEqual(report["descriptors"], 178)
        self.assertFalse(report["payload_interpreted"])
        self.assertEqual(report["rank"], 16)
        readout = (HERE / self.recipe["origin_checkpoint_file"]).read_bytes()
        head = check_checkpoint(readout, READOUT, self.recipe["source_manifest_sha256"])
        self.assertEqual(head["descriptors"], 2)
        self.assertFalse(policy["reexpand"])

    def test_exact_selected_epoch_and_producer_bind_real_reports(self):
        producer = json.loads((HERE / "recovery/rank16-selected-producer.json").read_text())
        selection = json.loads((HERE / "recovery/rank16-selection.json").read_text())
        self.assertEqual(producer["run_id"], 44762535076)
        self.assertEqual(producer["commit"], COMMIT)
        self.assertEqual(selection["selected_epoch"], 0)
        self.assertTrue(selection["epochs"][1]["selected"])
        self.assertFalse(selection["audit_used_for_selection"])
        self.assertFalse(producer["validation_gate"]["passed"])

    def test_resume_cannot_reroll_reexpand_or_use_audit(self):
        for key, value in (("reexpand", True), ("selected_epoch", 2), ("producer_run_id", 1),
                           ("producer_commit", "f" * 40), ("direct_load", False), ("audit_used", True)):
            bad = copy.deepcopy(self.recipe)
            bad["rank16_resume"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                resume_policy(bad)

    def test_swapped_checkpoint_or_readout_is_refused(self):
        for key in ("checkpoint_sha256", "origin_checkpoint_sha256"):
            bad = copy.deepcopy(self.recipe)
            bad[key] = "f" * 64
            with self.assertRaises(ValueError):
                resume_policy(bad)
        raw = (HERE / self.recipe["checkpoint_file"]).read_bytes()
        with self.assertRaises(ValueError):
            source_resume(raw + b"changed", self.recipe)

    def test_lower_rate_does_not_change_data_or_quality_gates(self):
        self.assertEqual(self.recipe["lr"], 1e-5)
        self.assertEqual(self.recipe["epochs"], 4)
        self.assertEqual(self.recipe["grad_accum"], 8)
        self.assertEqual(self.recipe["supplement_preferences"]["new_preference_rows"], 128)
        self.assertEqual(self.recipe["bare_loss_weight"], 16)
        bad = copy.deepcopy(self.recipe)
        bad["lr"] = 3e-5
        with self.assertRaises(ValueError):
            resume_policy(bad)
