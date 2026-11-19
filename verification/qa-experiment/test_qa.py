# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Artificial schema fixtures test gates only. They never become training data."""
import copy
import json
import os
import random
import struct
import tempfile
import unittest
import zipfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from data import (
    Components,
    check_splits,
    eligible,
    group_messages,
    norm,
    preferences,
    select,
)
from licensing import (
    OWN_ID,
    licence_metadata,
    preserve_artifact,
    tensor_parts,
    verify_checkpoint_metadata,
)
from live_class import CHILD, FAMILY, URLS, validate_snapshot
from run import checked_preparation_ast, quality, remote_only
from runtime_check import check_response, decode_bits, native_agreement
from training_policy import development_selection, isolated_evaluation


def message(ident, tree, text, parent=None, rank=None):
    return {"message_id": ident, "message_tree_id": tree, "text": text, "parent_id": parent,
            "rank": rank, "role": "prompter" if parent is None else "assistant", "lang": "en",
            "synthetic": False, "deleted": False, "review_count": 1, "review_result": True,
            "labels": {"name": ["pii"], "value": [0.]}}


class SourceGates(unittest.TestCase):
    def test_explicit_source_flags(self):
        row = message("r", "t", "root")
        self.assertTrue(eligible(row))
        for key, value in (("synthetic", True), ("synthetic", None), ("deleted", True),
                           ("deleted", None), ("review_result", False), ("review_count", 0),
                           ("labels", {"name": ["pii"], "value": [0.1]}), ("labels", [])):
            with self.subTest(key=key, value=value):
                bad = {**row, key: value}
                self.assertFalse(eligible(bad))

    def test_normalization(self):
        self.assertEqual(norm(" Ａ\tB\u0085C "), "a b c")

    def test_transitive_connectivity_actual_inputs(self):
        a = message("a", "tree-a", "first")
        b = message("b", "tree-b", " FIRST ")
        c = message("c", "tree-b", "bridge")
        d = message("d", "tree-c", "BRIDGE")
        d["synthetic"] = True
        candidates = [{"tree_id": "tree-a", "prompt": "first", "chosen": "a1", "rejected": "a2"},
                      {"tree_id": "tree-b", "prompt": " FIRST ", "chosen": "bridge", "rejected": "b2"},
                      {"tree_id": "tree-c", "prompt": "c", "chosen": "BRIDGE", "rejected": "c2"}]
        links, cross = group_messages({"train": [a, b, c], "validation": [d]}, candidates)
        self.assertEqual(links.find("tree-a"), links.find("tree-c"))
        self.assertEqual(cross, {links.find("tree-a")})

    def test_unused_descendants_never_merge_roots(self):
        roots = [message("a", "tree-a", "q-a"), message("b", "tree-b", "q-b")]
        descendants = [message("a-child", "tree-a", "thanks", "a"), message("b-child", "tree-b", "THANKS", "b")]
        candidates = [{"tree_id": "tree-a", "prompt": "q-a", "chosen": "answer-a", "rejected": "other-a"},
                      {"tree_id": "tree-b", "prompt": "q-b", "chosen": "answer-b", "rejected": "other-b"}]
        links, cross = group_messages({"train": [roots[0], descendants[0]], "validation": [roots[1], descendants[1]]}, candidates)
        self.assertNotEqual(links.find("tree-a"), links.find("tree-b"))
        self.assertFalse(cross)

    def test_rank_direction_and_token_visibility(self):
        root = message("r", "tree", "question")
        good = message("a", "tree", "preferred", "r", 1)
        bad = message("b", "tree", "alternative", "r", 2)
        pools, _ = preferences({"train": [root, bad, good], "validation": []}, lambda p, a: [ord(c) for c in a])
        rows = [r for values in pools.values() for r in values]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["chosen"], "preferred")
        empty, _ = preferences({"train": [root, bad, good]}, lambda p, a: [1] * 257)
        self.assertFalse(empty)
        collapsed, _ = preferences({"train": [root, bad, good]}, lambda p, a: [1])
        self.assertFalse(collapsed)

    def test_components_no_recursion_ceiling(self):
        links = Components()
        for i in range(3000, 0, -1):
            links.union(str(i), str(i - 1))
        self.assertEqual(links.find("3000"), links.find("0"))

    def test_split_leakage_fails(self):
        row = {"prompt": "q", "chosen": "a", "rejected": "b", "group_id": "g", "chosen_rank": 0, "rejected_rank": 1}
        other = copy.deepcopy(row)
        other["group_id"] = "other"
        with self.assertRaises(ValueError):
            check_splits({"train": [row], "audit": [other]})

    def test_insufficient_data_never_relaxes(self):
        from collections import defaultdict
        with self.assertRaises(ValueError):
            select(defaultdict(list), {"train": 128, "validation": 32, "audit": 64})

    def test_remote_guard_and_quality_control(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            remote_only()
        trained = {"pairwise_accuracy": .9, "subsets": {"shorter": .8, "longer": .9}}
        self.assertTrue(quality(trained, {"pairwise_accuracy": .6}, {"length": .5})["passed"])
        self.assertFalse(quality(trained, {"pairwise_accuracy": .9}, {"length": .5})["passed"])
        self.assertFalse(quality(trained, {"pairwise_accuracy": .6}, {"length": .9})["passed"])
        self.assertFalse(quality({**trained, "pairwise_accuracy": float("nan")}, {"pairwise_accuracy": .6}, {"length": .5})["passed"])

    def test_live_class_primary_identity_and_staleness(self):
        now = datetime.now(timezone.utc)
        value = {"schema": 1, "fetched_at": now.isoformat(),
                 "family": {"url": URLS["family"], "raw": json.dumps({"resolve": {"intent": FAMILY, "class": "judgment"}})},
                 "child": {"url": URLS["child"], "raw": json.dumps({"intent_id": CHILD, "intent_name": CHILD,
                           "class": "judgment", "evaluation": {"kind": "judgment"}, "request": None, "response": None})}}
        self.assertEqual(validate_snapshot(value, now), "judgment")
        bad = copy.deepcopy(value)
        bad["fetched_at"] = (now - timedelta(hours=1)).isoformat()
        with self.assertRaises(ValueError):
            validate_snapshot(bad, now)
        bad = copy.deepcopy(value)
        bad["family"]["url"] += "&override=true"
        with self.assertRaises(ValueError):
            validate_snapshot(bad, now)

    def test_runtime_identity_nonfinite_and_numerical_controls(self):
        from runtime_check import RELEASE_HASHES
        manifest = {"keccak256": "0x" + "a" * 64}
        response = {"foundation_hash": RELEASE_HASHES["foundation_lora_fp32.onnx"],
                    "tokenizer_hash": RELEASE_HASHES["tokenizer.json"], "adapter_hash": "a" * 64,
                    "results": [{"score": 0x3f800000}]}
        self.assertEqual(check_response(response, 1, manifest), [0x3f800000])
        with self.assertRaises(ValueError):
            check_response({**response, "adapter_hash": "b" * 64}, 1, manifest)
        with self.assertRaises(ValueError):
            decode_bits(0x7fc00000)
        self.assertTrue(native_agreement([1., 0.] * 64, [0x3f800000, 0] * 64)["ordering"])
        self.assertFalse(native_agreement([1., 0.] * 64, [0, 0x3f800000] * 64)["ordering"])

    def test_development_loss_selects_best_of_three_without_audit(self):
        best, chosen = development_selection(None, 1, {"val_loss": .6, "audit_accuracy": 1.})
        self.assertTrue(chosen)
        best, chosen = development_selection(best, 2, {"val_loss": .4, "audit_accuracy": 0.})
        self.assertTrue(chosen)
        best, chosen = development_selection(best, 3, {"val_loss": .4, "audit_accuracy": 1.})
        self.assertFalse(chosen)
        self.assertEqual(best, {"epoch": 2, "val_loss": .4})
        with self.assertRaises(ValueError):
            development_selection(best, 4, {"val_loss": 0.})
        with self.assertRaises(ValueError):
            development_selection(None, 1, {"val_loss": float("nan")})

    def test_evaluation_rng_context_restores_state_and_mode_on_error(self):
        # Source-only boundary test. Actual Torch/NumPy state equality is also
        # checked around each real development evaluation on the remote runner.
        numpy_rng, torch_rng = random.Random(4), random.Random(5)
        calls = []
        @contextmanager
        def fork_rng(*, devices, enabled):
            calls.append((devices, enabled))
            state = torch_rng.getstate()
            try:
                yield
            finally:
                torch_rng.setstate(state)
        class Mode:
            training = True
            def eval(self):
                self.training = False
            def train(self, state):
                self.training = state
        model = Mode()
        numpy = SimpleNamespace(random=SimpleNamespace(get_state=numpy_rng.getstate, set_state=numpy_rng.setstate))
        torch = SimpleNamespace(random=SimpleNamespace(fork_rng=fork_rng))
        before = random.getstate(), numpy_rng.getstate(), torch_rng.getstate()
        with self.assertRaises(ValueError), isolated_evaluation(model, torch, numpy):
            self.assertFalse(model.training)
            random.random()
            numpy_rng.random()
            torch_rng.random()
            raise ValueError("evaluation failed")
        self.assertEqual(before, (random.getstate(), numpy_rng.getstate(), torch_rng.getstate()))
        self.assertTrue(model.training)
        self.assertEqual(calls, [([], True)])

    def test_checkpoint_metadata_separates_terms_without_changing_payload(self):
        base = Path(__file__).resolve().parent / "evidence/tooling"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temp:
            path = Path(temp) / "toy.safetensors"
            header = {"__metadata__": {"format": "pt"},
                      "toy": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
            encoded = json.dumps(header).encode()
            raw = struct.pack("<Q", len(encoded)) + encoded + b"\0\0\0\0"
            path.write_bytes(raw)
            preserve_artifact(path)
            after, payload = tensor_parts(path.read_bytes())
            self.assertEqual(payload, b"\0\0\0\0")
            self.assertEqual(after["toy"], header["toy"])
            self.assertEqual(after["__metadata__"]["x-licence"], OWN_ID)
            self.assertEqual(after["__metadata__"]["x-own-sand"], licence_metadata()["x-own-sand"])
            self.assertEqual(after["__metadata__"]["x-upstream-apache-2.0"], licence_metadata()["x-upstream-apache-2.0"])
            once = path.read_bytes()
            preserve_artifact(path)
            self.assertEqual(path.read_bytes(), once)
            verify_checkpoint_metadata(path)

    def test_preparation_source_hash_gate_before_ast_reuse(self):
        source = (Path(__file__).resolve().parent.parent / "training/run.py").read_bytes()
        module = checked_preparation_ast(source)
        self.assertEqual(len(module.body), 1)
        self.assertEqual(module.body[0].name, "prepare")
        with self.assertRaises(ValueError):
            checked_preparation_ast(source + b"\n# modified preparation\n")
        with self.assertRaises(ValueError):
            checked_preparation_ast(b"this is not valid Python")

    def test_wrong_checkpoint_json_shape_is_malformed_container_value_error(self):
        for value in ([], {"__metadata__": []}):
            encoded = json.dumps(value).encode()
            raw = struct.pack("<Q", len(encoded)) + encoded
            with self.assertRaises(ValueError):
                tensor_parts(raw)

    def test_torch_zip_inert_licence_record_preserves_original_members(self):
        base = Path(__file__).resolve().parent / "evidence/tooling"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temp:
            path = Path(temp) / "toy.pt"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("toy/data.pkl", b"schema-only fixture, never unpickled")
                archive.writestr("toy/data/0", b"\0\0\0\0")
                archive.writestr("toy/version", b"3\n")
            preserve_artifact(path)
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(archive.read("toy/data/0"), b"\0\0\0\0")
                self.assertEqual(json.loads(archive.read("toy/qa-licences.json")), licence_metadata())
            once = path.read_bytes()
            preserve_artifact(path)
            self.assertEqual(path.read_bytes(), once)
            verify_checkpoint_metadata(path)


if __name__ == "__main__":
    unittest.main()
