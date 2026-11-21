# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import sys
import unittest
from unittest.mock import patch

import data
from expanded_data import extend
from runtime_adapt import adaptation_parameter, select_epoch, train_adapted_head


def synthetic(i):
    return {
        "prompt": f"synthetic prompt {i}",
        "chosen": f"first synthetic {i}",
        "rejected": f"second synthetic {i}",
        "group_id": data.sha(str(i).encode()),
    }


class ExpandedAdaptation(unittest.TestCase):
    def test_reserved_bytes_and_order_preserved(self):
        pool = [synthetic(i) for i in range(8)]
        old = {"train": pool[:2], "validation": pool[2:4], "audit": pool[4:6]}
        before = copy.deepcopy(old)
        expanded = extend(pool, old, extra=2)
        self.assertEqual(old, before)
        self.assertIs(expanded["validation"], old["validation"])
        self.assertIs(expanded["audit"], old["audit"])
        self.assertEqual(expanded["train"], pool[:2] + pool[6:])

    def test_overlap_insufficient_and_padding_refuse(self):
        pool = [synthetic(i) for i in range(7)]
        old = {"train": pool[:2], "validation": pool[2:4], "audit": pool[4:6]}
        with self.assertRaises(ValueError):
            extend(pool, old, extra=2)
        for bad in (True, 0, 513):
            with self.assertRaises(ValueError):
                extend(pool, old, extra=bad)
        pool[-1]["chosen"] = old["audit"][0]["chosen"]
        with self.assertRaises(ValueError):
            extend(pool, old, extra=1)

    def test_literal_supported_parameter_boundary(self):
        prefix = "base_model.model.model.layers."
        self.assertEqual(
            adaptation_parameter(prefix + "21.attn.Wqkv.lora_B.default.weight", 22),
            "lora",
        )
        self.assertIsNone(
            adaptation_parameter(prefix + "17.attn.Wqkv.lora_B.default.weight", 22)
        )
        self.assertIsNone(
            adaptation_parameter(prefix + "22.attn.Wqkv.lora_B.default.weight", 22)
        )
        self.assertIsNone(
            adaptation_parameter("base_model.model.head.dense.weight", 22)
        )
        self.assertIsNone(
            adaptation_parameter(prefix + "21.attn.Wqkv.base_layer.weight", 22)
        )
        self.assertIsNone(
            adaptation_parameter(prefix + "21.attn.Wqkv.lora_B.default.bias", 22)
        )

    def test_complete_selection_prefers_unchanged_control_on_tie(self):
        rows = [
            {"epoch": i, "validation": {"pairs": 64, "accuracy": 0.6}} for i in range(3)
        ]
        self.assertEqual(select_epoch(rows), 0)
        rows[2]["validation"]["accuracy"] = 0.7
        self.assertEqual(select_epoch(rows), 2)
        for bad in (rows[:2], list(reversed(rows))):
            with self.assertRaises(ValueError):
                select_epoch(bad)
        rows[0]["validation"]["accuracy"] = float("nan")
        with self.assertRaises(ValueError):
            select_epoch(rows)

    def test_neural_work_refused_before_import(self):
        with patch.dict("os.environ", {"GITHUB_ACTIONS": "false"}):
            with self.assertRaises(RuntimeError):
                train_adapted_head(*([None] * 9))
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)


if __name__ == "__main__":
    unittest.main()
