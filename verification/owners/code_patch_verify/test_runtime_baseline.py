# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Pure source wire controls. No model is created, loaded, trained or inferred."""
import copy
import unittest
from unittest.mock import patch

from runtime_baseline import ASSETS, check_spec, input_root, response_bits, write_control_package


class BaselineSourceTests(unittest.TestCase):
    def setUp(self):
        self.identity = {"foundation_hash": "a" * 64, "adapter_hash": "b" * 64, "tokenizer_hash": "c" * 64}
        self.reply = {**self.identity, "input_root": input_root([[1, 2]]), "results": [{"score": 0}]}

    def test_literal_wire_shape(self):
        self.assertEqual(response_bits(self.reply, self.identity, self.reply["input_root"], 1), [0])

    def test_identity_root_and_count_refusals(self):
        for key, value in (("adapter_hash", "0x" + "b" * 64), ("input_root", "different"), ("results", [])):
            changed = copy.deepcopy(self.reply)
            changed[key] = value
            with self.assertRaises(ValueError):
                response_bits(changed, self.identity, self.reply["input_root"], 1)

    def test_bit_conflict_and_nonfinite_controls(self):
        for result in ({"score": 0, "error": "conflict"}, {"score": True}, {"score": 0x7f800000}, {"score": -1}):
            changed = copy.deepcopy(self.reply)
            changed["results"] = [result]
            with self.assertRaises(ValueError):
                response_bits(changed, self.identity, self.reply["input_root"], 1)

    def test_input_root_literal_native_tokens(self):
        self.assertNotEqual(input_root([[1, 2]]), input_root([[1], [2]]))
        for batch in ([[]], [[True]], [[-1]], [[50368]], [[1] * 513]):
            with self.assertRaises(ValueError):
                input_root(batch)

    def test_no_local_control_package_creation(self):
        with patch.dict("os.environ", {"GITHUB_ACTIONS": "false"}):
            with self.assertRaises(RuntimeError):
                write_control_package(None, {})

    def test_bounded_released_layout(self):
        spec = {"foundation_hash": ASSETS["foundation_lora_fp32.onnx"][0], "tokenizer_hash": ASSETS["tokenizer.json"][0],
                "foundation_version": "modernbert-base-lora16-v1", "r_max": 16, "num_layers": 22, "hidden_size": 768,
                "targets": {"attn.Wqkv": [768, 2304], "attn.Wo": [768, 768], "mlp.Wi": [768, 2304], "mlp.Wo": [1152, 768]}}
        check_spec(spec)
        spec["num_layers"] = True
        with self.assertRaises(ValueError):
            check_spec(spec)


if __name__ == "__main__":
    unittest.main()
