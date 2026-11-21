# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Offline true-data/schedule and failure controls. No model operations."""
import copy
import json
import os
import random
import unittest
from collections import Counter
from unittest.mock import patch

from common import HERE, FORMATS
from transfer_contract import plan, schedule, transfer_policy
from test_gradient_diagnostic import training_population
from checkpoint_contract import check_checkpoint


class TransferTests(unittest.TestCase):
    def test_actual_population_and_all_existing_bare_views_are_retained(self):
        groups, binding = plan(training_population(), json.loads((HERE / "recipe.json").read_text()))
        self.assertEqual(binding, json.loads((HERE / "transfer-plan.json").read_text()))
        self.assertEqual(binding["unique_training_rows"], 1008)
        self.assertEqual(binding["unique_replayed_rows"], 48)
        self.assertEqual(binding["independent_replayed_groups"], 24)
        self.assertEqual({s: len(g) for s, g in groups.items()}, {s: 8 for s in FORMATS})

    def test_schedule_balances_formats_origins_and_round_coverage(self):
        recipe = json.loads((HERE / "recipe.json").read_text())
        groups, binding = plan(training_population(), recipe)
        before = random.getstate()
        blocks = schedule(groups, 0, binding["policy"])
        self.assertEqual(before, random.getstate())
        self.assertEqual(blocks, schedule(groups, 0, binding["policy"]))
        self.assertEqual(len(blocks), 24)
        for block in blocks:
            rows = [r for g in block["groups"] for r in g]
            self.assertEqual(len(rows), 8)
            self.assertEqual(sorted(Counter(g[0]["style"] for g in block["groups"]).values()), [1, 1, 2])
            self.assertEqual(sum(r["label"] == "Human" for r in rows), 4)
            self.assertEqual(sum(r["label"] == "AI-generated" for r in rows), 4)
        for round_index in range(4):
            self.assertEqual(len({g[0]["view_id"] for b in blocks if b["round"] == round_index for g in b["groups"]}), 24)

    def test_policy_cannot_replay_audit_or_expand_budget(self):
        recipe = json.loads((HERE / "recipe.json").read_text())
        for key, value in (("rounds_per_epoch", 8), ("replay_pair_weight", 16.), ("validation_or_audit_replayed", True), ("runtime_transform", True)):
            bad = copy.deepcopy(recipe)
            bad["format_transfer_curriculum"][key] = value
            with self.assertRaises(ValueError):
                transfer_policy(bad)

    def test_validation_role_or_population_mutation_is_refused(self):
        rows = training_population()
        recipe = json.loads((HERE / "recipe.json").read_text())
        for key, value in (("split", "validation"), ("group_id", "f" * 64), ("label", "Mixed")):
            bad = copy.deepcopy(rows)
            bad[0][key] = value
            with self.assertRaises(ValueError):
                plan(bad, recipe)
        with self.assertRaises(ValueError):
            plan(rows[:-2], recipe)

    def test_neural_transfer_refuses_local_execution(self):
        import train_transfer
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                train_transfer.replay(*([None] * 11))

    def test_actual_preserved_reference_grants_hashes_and_tamper_refusal(self):
        reference = json.loads((HERE / "transfer-reference.json").read_text())
        recipe = json.loads((HERE / "recipe.json").read_text())
        self.assertEqual(reference["producer_run_id"], 44860695142)
        self.assertEqual(reference["selected_epoch"], 3)
        self.assertIs(reference["quality_gate_passed"], False)
        self.assertIs(reference["registration_eligible"], False)
        self.assertIs(reference["audit_used"], False)
        for prefix, count in (("checkpoint", 178), ("readout", 2)):
            path = HERE / reference[prefix + "_file"]
            self.assertTrue(path.resolve().is_relative_to(HERE.resolve()))
            raw = path.read_bytes()
            checked = check_checkpoint(raw, reference[prefix + "_sha256"], recipe["source_manifest_sha256"])
            self.assertEqual(checked["descriptors"], count)
            with self.assertRaises(ValueError):
                check_checkpoint(raw[:-1] + bytes([raw[-1] ^ 1]), reference[prefix + "_sha256"], recipe["source_manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
