# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import unittest

from warmstart import package_tensor


class WarmStartTests(unittest.TestCase):
    def test_inverse_primary_export_names_and_transposes(self):
        self.assertEqual(package_tensor("base_model.model.model.layers.3.attn.Wqkv.lora_A.default.weight"), ("lora.3.attn.Wqkv.A", True))
        self.assertEqual(package_tensor("base_model.model.model.layers.3.mlp.Wo.lora_B.default.weight"), ("lora.3.mlp.Wo.B", True))
        self.assertEqual(package_tensor("base_model.model.classifier.modules_to_save.default.weight"), ("head.weight", True))
        self.assertEqual(package_tensor("base_model.model.classifier.modules_to_save.default.bias"), ("head.bias", False))

    def test_frozen_base_and_original_head_are_not_overwritten(self):
        self.assertIsNone(package_tensor("base_model.model.model.layers.3.attn.Wqkv.base_layer.weight"))
        self.assertIsNone(package_tensor("base_model.model.classifier.original_module.weight"))
        self.assertIsNone(package_tensor("base_model.model.model.layers.3.other.lora_A.default.weight"))


if __name__ == "__main__":
    unittest.main()
