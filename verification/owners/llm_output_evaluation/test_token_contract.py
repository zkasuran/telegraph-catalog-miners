# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Lexical contract controls only, no neural models or source/audit scores."""
from types import SimpleNamespace
import unittest
from token_contract import checked_tokens


class TokenContractTests(unittest.TestCase):
    def encoders(self, native, node):
        class Native:
            def __call__(self, text, **kwargs):
                self.text, self.options = text, kwargs
                return {"input_ids": native}
        class Node:
            def encode(self, text, **kwargs):
                self.text, self.options = text, kwargs
                return SimpleNamespace(ids=node)
        return Native(), Node()

    def test_literal_input_and_options(self):
        native, node = self.encoders([50281, 22, 50282], [50281, 22, 50282])
        text = '### Task:\n<&"\nΩ\n\n### Answer:\nPoor'
        result = checked_tokens(text, native, node)
        self.assertEqual(result, [50281, 22, 50282])
        self.assertEqual(native.text, text)
        self.assertEqual(node.text, text)
        self.assertEqual(native.options, {"truncation": False, "padding": False, "add_special_tokens": True})
        self.assertEqual(node.options, {"add_special_tokens": True})

    def test_different_tokens_refused(self):
        native, node = self.encoders([50281, 22, 50282], [50281, 23, 50282])
        with self.assertRaisesRegex(ValueError, "differs"):
            checked_tokens("literal", native, node)

    def test_padding_is_not_normalized_away(self):
        native, node = self.encoders([50281, 22, 50282, 50283], [50281, 22, 50282])
        with self.assertRaises(ValueError):
            checked_tokens("literal", native, node)

    def test_special_tokens_and_ids_refused(self):
        for ids in ([22, 50282], [50281, 22], [50281, True, 50282], [50281, -1, 50282], [50281, 50368, 50282]):
            native, node = self.encoders(ids, ids)
            with self.assertRaises(ValueError):
                checked_tokens("literal", native, node)

    def test_overflow_refused_not_cropped(self):
        ids = [50281] + [22] * 255 + [50282]
        native, node = self.encoders(ids, ids)
        with self.assertRaisesRegex(ValueError, "whole rendered"):
            checked_tokens("literal", native, node)
        self.assertEqual(len(ids), 257)

    def test_cap_type_and_text_ceiling(self):
        native, node = self.encoders([50281, 22, 50282], [50281, 22, 50282])
        for cap in (True, 1, 513, "256"):
            with self.assertRaises(ValueError):
                checked_tokens("literal", native, node, cap)
        with self.assertRaises(ValueError):
            checked_tokens("x" * ((1 << 20) + 1), native, node)


if __name__ == "__main__":
    unittest.main()
