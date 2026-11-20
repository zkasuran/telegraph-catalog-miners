# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Opaque descriptor and policy controls. No local model operations."""
import copy
import json
import os
import struct
import unittest
from unittest.mock import patch

from common import HERE, sha
from capacity_contract import capacity_policy, source_capacity


class CapacityTests(unittest.TestCase):
    def setUp(self):
        # Keep every original rank8 expansion check on its actual historical recipe.
        # The active rank16 direct-resume contract has its own additional tests.
        self.recipe = json.loads((HERE / "recovery/rank16-selected-producer.json").read_text())["checkpoint_lineage"]

    def test_exact_supported_capacity_and_scale(self):
        policy = capacity_policy(self.recipe)
        self.assertEqual(policy["new_rank"], 16)
        self.assertEqual(policy["old_alpha"] / policy["old_rank"], policy["new_alpha"] / policy["new_rank"])
        self.assertFalse(policy["runtime_transform"])
        self.assertFalse(policy["audit_used"])

    def test_changed_capacity_scale_or_retention_is_refused(self):
        for key, value in (("new_rank", 32), ("new_alpha", 32), ("old_rank", 16),
                           ("new_b", "random"), ("head_retained", False), ("runtime_transform", True), ("audit_used", True)):
            bad = copy.deepcopy(self.recipe)
            bad["capacity_expansion"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                capacity_policy(bad)
        bad = copy.deepcopy(self.recipe)
        bad["lora_rank"] = 16.0
        with self.assertRaises(ValueError):
            capacity_policy(bad)

    def test_real_original_checkpoint_is_opaque_and_still_rank8(self):
        raw = (HERE / self.recipe["checkpoint_file"]).read_bytes()
        report = source_capacity(raw, self.recipe)
        self.assertEqual(report["original_lora_descriptors"], 176)
        self.assertEqual(report["retained_head_descriptors"], 2)
        self.assertFalse(report["tensor_payload_interpreted"])
        self.assertFalse(report["expanded_state_built"])
        self.assertEqual(report["sha256"], "e426a298d84b73c0b3a853e51695d8a8d6f77a7d5f65eb24a1b222e2f70d85ef")

    def test_changed_original_descriptors_are_refused_even_with_fixture_hash(self):
        raw = (HERE / self.recipe["checkpoint_file"]).read_bytes()
        n = struct.unpack("<Q", raw[:8])[0]
        head = json.loads(raw[8:8 + n])
        name = next(k for k in head if ".lora_A." in k)
        head[name]["shape"][0] = 16
        encoded = json.dumps(head).encode()
        fixture = struct.pack("<Q", len(encoded)) + encoded + raw[8 + n:]
        recipe = {**self.recipe, "checkpoint_sha256": sha(fixture)}
        with self.assertRaisesRegex(ValueError, "rank8 descriptors"):
            source_capacity(fixture, recipe)

    def test_expansion_and_tensor_controls_refuse_local_execution(self):
        import train_capacity
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                train_capacity.expand(None, None, None)
            with self.assertRaises(RuntimeError):
                train_capacity.smoke(None, None)

    def test_retained_scale_formula_on_explicit_scalar_fixture(self):
        # Synthetic arithmetic fixture, not checkpoint tensor bytes or inference.
        a, b = list(range(1, 9)), list(range(2, 10))
        new_a, new_b = a + list(range(11, 19)), b + [0] * 8
        old_delta = 4 * sum(x * y for x, y in zip(a, b))
        new_delta = 4 * sum(x * y for x, y in zip(new_a, new_b))
        self.assertEqual(old_delta, new_delta)
        new_b[8] = 1
        self.assertNotEqual(old_delta, 4 * sum(x * y for x, y in zip(new_a, new_b)))

    def test_capacity_does_not_change_data_or_quality_policy(self):
        from source_preferences import preference_policy
        self.assertEqual(preference_policy(self.recipe)["original_preference_rows"], 752)
        self.assertEqual(preference_policy(self.recipe)["new_preference_rows"], 128)
        self.assertEqual(self.recipe["epochs"], 4)
        self.assertEqual(self.recipe["grad_accum"], 8)
        self.assertEqual(self.recipe["lr"], 3e-05)
        self.assertEqual(self.recipe["compact_training"]["heldout_changed"], False)


if __name__ == "__main__":
    unittest.main()
