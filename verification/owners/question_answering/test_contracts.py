# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Source-only boundary controls. Artificial fixtures never enter the corpus."""
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from data import RECIPE, choose, prepare_audit
from frozen_baseline import decode_frozen
from licensing import licence_metadata
from runtime_check import check_response, decode_bits, model_remote_only
from source_data import check_splits, eligible, preferences
from training_policy import development_selection


class Contracts(unittest.TestCase):
    def test_exposed_or_shared_source_components_cannot_cross_splits(self):
        row = {"group_id": "a", "tree_id": "tree", "prompt": "source question", "chosen": "source preferred",
               "rejected": "source alternative", "chosen_rank": 0, "rejected_rank": 1}
        with self.assertRaises(ValueError):
            check_splits({"train": [row], "audit": [{**row, "group_id": "b"}]})

    def test_source_flags_and_clipped_preferences_refused(self):
        root = {"message_id": "root", "message_tree_id": "tree", "parent_id": None, "text": "question",
                "role": "prompter", "lang": "en", "synthetic": False, "deleted": False, "review_result": True,
                "review_count": 1, "labels": {"name": ["pii"], "value": [0]}, "rank": None}
        self.assertTrue(eligible(root))
        self.assertFalse(eligible({**root, "synthetic": True}))
        self.assertFalse(eligible({**root, "labels": {"name": ["pii"], "value": [.1]}}))
        answers = [{**root, "message_id": name, "parent_id": "root", "role": "assistant", "text": name, "rank": rank}
                   for name, rank in (("preferred", 0), ("alternative", 1))]
        pools, _ = preferences({"train": [root, *answers]}, lambda p, a: [1] * 257)
        self.assertFalse(pools)

    def test_remote_guard_ignores_ci_fixture_environment(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            model_remote_only()

    def test_fixed_recipe_and_development_selection(self):
        self.assertEqual(RECIPE["epochs"], 3)
        best, _ = development_selection(None, 1, {"val_loss": .7})
        best, _ = development_selection(best, 2, {"val_loss": .4})
        best, selected = development_selection(best, 3, {"val_loss": .5})
        self.assertFalse(selected)
        self.assertEqual(best["epoch"], 2)

    def test_source_selection_cannot_lower_counts(self):
        with self.assertRaises(ValueError):
            choose([], 64, "artificial-fixture")

    def test_audit_refuses_missing_development_freeze(self):
        with self.assertRaises(FileNotFoundError):
            prepare_audit(Path("missing-source-only-fixture"))

    def test_full_own_and_upstream_grants_are_separate(self):
        value = licence_metadata()
        self.assertIn("Source-Available", value["x-own-sand"])
        self.assertIn("Apache License", value["x-upstream-apache-2.0"])
        self.assertIn("MIT License", value["x-upstream-mit"])

    def test_duplicate_and_nonfinite_controls(self):
        with self.assertRaises(ValueError):
            decode_frozen(b'{"fit_split":"train","fit_split":"audit"}')
        with self.assertRaises(ValueError):
            decode_bits(0x7fc00000)

    def test_literal_wire_echo_not_normalized_standin(self):
        from runtime_check import RELEASE_HASHES
        manifest = {"keccak256": "0x" + "a" * 64}
        value = {"foundation_hash": RELEASE_HASHES["foundation_lora_fp32.onnx"],
                 "tokenizer_hash": RELEASE_HASHES["tokenizer.json"], "adapter_hash": "a" * 64,
                 "results": [{"score": 0x3f800000}]}
        self.assertEqual(check_response(value, 1, manifest), [0x3f800000])
        with self.assertRaises(ValueError):
            check_response({**value, "adapter_hash": "0x" + "a" * 64}, 1, manifest)


if __name__ == "__main__":
    unittest.main()
