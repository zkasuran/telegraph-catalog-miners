# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import json
import os
import random
import unittest
from unittest.mock import patch

from build import make_block, node_facts, python_facts, render, validate
from remote import check_reply, remote_only
from scorer import score


class SourceTests(unittest.TestCase):
    def test_direct_python_addition(self):
        text = ".. function:: alpha(x)\n\n   Text.\n\n   .. versionadded:: 3.10\n"
        self.assertEqual(python_facts(text)[0]["version"], "3.10")

    def test_parameter_addition_is_not_api_addition(self):
        text = ".. function:: alpha(x)\n\n   .. versionadded:: 3.10\n      Added the x parameter.\n"
        self.assertEqual(python_facts(text), [])

    def test_nested_method_does_not_label_parent(self):
        text = ".. class:: Parent()\n\n   .. method:: child(x)\n\n      .. versionadded:: 3.10\n"
        self.assertEqual([r["entity"] for r in python_facts(text)], ["child"])

    def test_node_scalar_not_multi_release(self):
        self.assertEqual(node_facts("## `util.alpha(x)`\n\n<!-- YAML\nadded: v8.2.0\n-->\n")[0]["version"], "v8.2.0")
        self.assertEqual(node_facts("## `util.alpha(x)`\n\n<!-- YAML\nadded:\n  - v8.2.0\n  - v9.0.0\n-->\n"), [])

    def block(self):
        return make_block("python", "Doc/library/example.rst", {"entity": "alpha", "version": "3.10"},
                          {"entity": "beta", "version": "3.12"}, {"revision": "a" * 40, "licence": "PSF-2.0"})

    def test_real_literal_body_primary_smoke(self):
        block = self.block()
        self.assertEqual(len(block), 2)
        for row in block:
            self.assertEqual(score(row["prompt"], row["chosen"]), 1)
            self.assertEqual(score(row["prompt"], row["rejected"]), 0)
            self.assertEqual(len(row["chosen"]), len(row["rejected"]))
            self.assertEqual(json.loads(row["chosen"])["sources"], json.loads(row["rejected"])["sources"])

    def test_evidence_removal_and_version_reversal(self):
        row = self.block()[0]
        query = json.loads(row["prompt"])
        query.pop("evidence")
        self.assertEqual(score(json.dumps(query), row["chosen"]), 0)
        query["evidence"] = "alpha: added 3.12; beta: added 3.10."
        self.assertEqual(score(json.dumps(query), row["chosen"]), 0)
        self.assertEqual(score(json.dumps(query), row["rejected"]), 1)

    def test_processed_brief_is_not_raw_body(self):
        row = self.block()[0]
        with self.assertRaises(ValueError):
            score(row["prompt"], json.dumps({"brief": json.loads(row["chosen"])["brief"]}))

    def test_group_leakage_cannot_pass(self):
        rows = [copy.deepcopy(r) for _ in range(32) for r in self.block()]
        with self.assertRaisesRegex(ValueError, "repeated negative"):
            validate({"train": rows})

    def test_normalized_overlap_cannot_pass(self):
        rows = []
        for i in range(32):
            rows.extend(make_block("python", f"Doc/library/example{i}.rst",
                        {"entity": f"alpha{i}", "version": "3.10"}, {"entity": f"beta{i}", "version": "3.12"},
                        {"revision": "a" * 40, "licence": "PSF-2.0"}))
        validate({"train": rows})
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate({"train": rows, "validation": copy.deepcopy(rows)})

    def test_literal_renderer_not_normalized(self):
        self.assertEqual(render("a\r\nb", " c "), "### Task:\na\r\nb\n\n### Answer:\n c ")
        self.assertNotEqual(render(" a", "b"), render("a", "b"))

    def test_random_untrusted_strings_bounded(self):
        rng = random.Random(42)
        for _ in range(200):
            text = "".join(chr(rng.randrange(128)) for _ in range(rng.randrange(200)))
            try:
                result = score(text, text)
            except ValueError:
                continue
            self.assertIn(result, (0.0, 1.0))
        with self.assertRaises(ValueError):
            score("x" * 5000, "{}")

    def test_remote_model_guard_refuses_laptop(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "bounded public CPU"):
                remote_only()

    def test_literal_runtime_identity_and_bits(self):
        manifest = {"keccak256": "0x" + "a" * 64}
        from common import FOUNDATION
        foundation = FOUNDATION
        from build import TOKENIZER_SHA
        reply = {"adapter_hash": "a" * 64, "foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
                 "tokenizer_hash": TOKENIZER_SHA, "results": [{"score": 1065353216}]}
        self.assertEqual(check_reply(reply, 1, manifest, foundation), [1065353216])
        changed = copy.deepcopy(reply)
        changed["adapter_hash"] = "A" * 64
        with self.assertRaisesRegex(ValueError, "identity"):
            check_reply(changed, 1, manifest, foundation)
        changed = copy.deepcopy(reply)
        changed["results"][0]["score"] = 2143289344
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            check_reply(changed, 1, manifest, foundation)
        with self.assertRaisesRegex(ValueError, "count"):
            check_reply(reply, 2, manifest, foundation)


if __name__ == "__main__":
    unittest.main()
