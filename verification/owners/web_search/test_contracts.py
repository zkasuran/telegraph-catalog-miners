# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import json
import unittest

from common import FOUNDATION, strict_json
from contracts import finite_bits, literal_reply, orientation, quality_gate, read_splits, source_gate
from licensing import tensor_parts


class Contracts(unittest.TestCase):
    def test_original_data_and_group_counts(self):
        recipe, _, splits = read_splits()
        self.assertEqual([len(splits[key]) for key in ("train", "validation", "audit")], [256, 64, 128])
        self.assertEqual(recipe["files"]["audit"]["pages"], 39)
        self.assertEqual(source_gate()["audit_metrics"], "sealed")

    def test_orientation_exact_not_suffix(self):
        row = {"prompt": json.dumps({"query": "Which Python version added za?"}), "facts": [{"entity": "a"}, {"entity": "za"}]}
        self.assertEqual(orientation(row), "target_second")

    def test_unchanged_quality_floors(self):
        measured = {"accuracy": .80, "per_label": {"first": .70, "second": .90}}
        self.assertTrue(quality_gate(measured, .70, .60))
        for field, changed in (("accuracy", .79), ("per_label", {"first": .69})):
            bad = copy.deepcopy(measured)
            bad[field] = changed
            self.assertFalse(quality_gate(bad, .50, .50))
        self.assertFalse(quality_gate(measured, .76, .50))
        self.assertFalse(quality_gate(measured, .50, .71))

    def test_literal_hash_echo_not_normalized(self):
        reply = {"adapter_hash": "a" * 64, "foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
                 "tokenizer_hash": FOUNDATION["tokenizer.json"], "results": [{"score": 1065353216}]}
        self.assertEqual(literal_reply(reply, 1, "a" * 64), [1065353216])
        for value in ("0x" + "a" * 64, "A" * 64, "a" * 64 + " "):
            changed = copy.deepcopy(reply)
            changed["adapter_hash"] = value
            with self.assertRaises(ValueError):
                literal_reply(changed, 1, "a" * 64)

    def test_nonfinite_bool_and_count_refused(self):
        for value in (True, -1, 1 << 32, 0x7fc00000, 0x7f800000):
            with self.assertRaises(ValueError):
                finite_bits(value)

    def test_duplicate_wire_field_refused(self):
        with self.assertRaises(ValueError):
            strict_json(b'{"adapter_hash":"a","adapter_hash":"b"}')

    def test_checkpoint_header_malformed_is_not_loaded(self):
        for raw in (b"", b"\0" * 8, b"\xff" * 8):
            with self.assertRaises(ValueError):
                tensor_parts(raw)


if __name__ == "__main__":
    unittest.main()
