# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only release tests. Constructed fixtures are never training data."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from common import (COUNTS, FORMATS, HERE, baseline_report, check_splits, fit_baselines,
                    identity, pair, quality, remote_only, render, sha)
from preserve import stamp
from reference_render import render as primary_render
from source import parse_article
from wire import check_reply, score_value


def fixture_splits():
    result = {}
    for split, n in COUNTS.items():
        result[split] = []
        for i in range(n):
            source = "fixture-" + split + "-" + str(i)
            year = 2014 if split == "train" else 2015 if split == "validation" else 2016
            for origin in ("human_control", "synthetic"):
                row = {"source_id": source, "group_id": identity(source), "origin": origin,
                       "text": f"Constructed parser fixture only {source} {origin} words for a source-only unit test.",
                       "split": split, "year": year, "generator": "distilgpt2" if split == "test" else "gpt2"}
                result[split].append(pair(row))
    return result


class SourceTests(unittest.TestCase):
    def test_full_manifest_counts_grants_groups_and_spans(self):
        source = json.loads((HERE / "sources.json").read_text())
        report = source["source_report"]
        self.assertFalse(report["inventory_truncated"])
        self.assertEqual(report["inventory_xml_files"], 32141)
        self.assertEqual(report["inventory_article_groups"], 19701)
        self.assertEqual(report["scanned"], 960)
        self.assertEqual(report["counts"], COUNTS)
        authors, texts, groups = set(), set(), set()
        for split, rows in report["articles"].items():
            self.assertEqual(len(rows), COUNTS[split])
            for row in rows:
                self.assertEqual(row["split"], split)
                self.assertEqual(row["group_id"], identity(row["source_id"]))
                self.assertNotIn(row["group_id"], groups)
                groups.add(row["group_id"])
                self.assertFalse(authors.intersection(row["author_ids"]))
                authors.update(row["author_ids"])
                for text in (row["paragraph"], row["prefix"], row["human_text"]):
                    self.assertNotIn(identity(text), texts)
                    texts.add(identity(text))
                start, end = row["human_span"]
                self.assertEqual(row["paragraph"][start:end], row["human_text"])
                self.assertEqual(len(row["human_text"].split()), 24)
                self.assertLessEqual(len(row["human_text"].encode()), 256)
                self.assertIn(row["licence"], ("CC-BY-3.0", "CC-BY-4.0"))
                self.assertIn("unrestricted", row["grant_clause"])
                self.assertIn("credited", row["grant_clause"])
                self.assertTrue(row["authors"])
                self.assertTrue(row["url"].endswith(row["path"]))
                self.assertIn("assumption", row["origin_basis"])
        for name, spec in source["generators"].items():
            self.assertEqual(spec["weight_sha256"], spec["files"]["model.safetensors"])
            self.assertEqual(len(spec["revision"]), 40)
            self.assertIn("tokenizer.json", spec["files"])
        for name, item in source["terms"].items():
            if name != "sand":
                self.assertEqual(sha(item["text"].encode()), item["sha256"])

    def test_literal_primary_render_including_whitespace_and_unicode(self):
        for prompt, answer in (("x", "y"), (" A\nB ", " \"Human\"\n"), ("é\u2028x", "<AI>&")):
            self.assertEqual(render(prompt, answer), primary_render(prompt, answer))
        self.assertEqual(render("x", "y"), "### Task:\nx\n\n### Answer:\ny")

    def test_live_verdict_spellings_and_same_cues(self):
        rows = fixture_splits()["train"][:2]
        a, b = [json.loads(r["chosen"]) for r in rows]
        self.assertEqual(a["verdict"], "Human")
        self.assertEqual(b["verdict"], "AI-generated")
        for row in rows:
            self.assertEqual(json.loads(row["chosen"])["cues"], json.loads(row["rejected"])["cues"])
            self.assertNotIn("ai_likelihood", json.loads(row["chosen"]))

    def test_valid_fixture_grouping(self):
        check_splits(fixture_splits())

    def test_wrong_group_and_time_fail(self):
        for field, value in (("group_id", "0" * 64), ("year", 2026)):
            splits = fixture_splits()
            splits["train"][0][field] = value
            with self.assertRaises(ValueError):
                check_splits(splits)

    def test_cross_split_source_alias_fails(self):
        splits = fixture_splits()
        for r in splits["test"][:2]:
            r["source_id"] = splits["train"][0]["source_id"]
            r["group_id"] = splits["train"][0]["group_id"]
        with self.assertRaisesRegex(ValueError, "cross-split"):
            check_splits(splits)

    def test_normalized_prompt_alias_fails(self):
        splits = fixture_splits()
        text = json.loads(splits["train"][0]["prompt"])["text"]
        splits["test"][0]["prompt"] = json.dumps({"text": "  " + text.upper() + "  "})
        splits["test"][0]["text_id"] = identity(text)
        with self.assertRaisesRegex(ValueError, "overlap"):
            check_splits(splits)

    def test_heldout_checkpoint_violation_fails(self):
        splits = fixture_splits()
        splits["test"][1]["generator"] = "gpt2"
        with self.assertRaisesRegex(ValueError, "held-out"):
            check_splits(splits)

    def test_duplicate_or_mislabeled_origin_fails(self):
        for field, value in (("label", "Mixed"), ("rejected", None)):
            splits = fixture_splits()
            splits["train"][0][field] = value if value else splits["train"][0]["chosen"]
            with self.assertRaises(ValueError):
                check_splits(splits)

    def test_integer_baselines_frozen_and_balanced(self):
        splits = fixture_splits()
        fitted = fit_baselines(splits["train"])
        before = json.dumps(fitted, sort_keys=True)
        self.assertTrue(all(type(n) is int for counts in fitted["integer_word_counts"].values() for n in counts.values()))
        report = baseline_report(splits["test"], fitted)
        self.assertEqual(report["length"], 0.5)
        self.assertEqual(report["answer_only"], 0.5)
        self.assertEqual(json.dumps(fitted, sort_keys=True), before)
        self.assertEqual(fit_baselines(splits["train"]), fitted)

    def test_local_remote_entrypoints_refuse_before_model_import(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "controller-owned"):
                remote_only()
            import generate
            import run
            import audit_runtime
            for main in (generate.main, run.main, audit_runtime.main):
                with self.assertRaises(RuntimeError):
                    main()

    def test_xml_tamper_and_missing_grant_refused(self):
        raw = b"<article><front><article-meta/></front></article>"
        entry = {"sha": "0" * 40, "size": len(raw), "path": "fake.xml"}
        with self.assertRaisesRegex(ValueError, "git tree"):
            parse_article(raw, entry)
        entry["sha"] = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        with self.assertRaises(ValueError):
            parse_article(raw, entry)

    def test_terms_stamp_preserves_actual_tensor_bytes(self):
        header = {"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
        h = json.dumps(header).encode()
        tensor = struct.pack("<f", 1.0)
        raw = struct.pack("<Q", len(h)) + h + tensor
        result = stamp(raw, {"sand": {"text": "unit-test notice"}})
        size = struct.unpack("<Q", result[:8])[0]
        self.assertEqual(result[8+size:], tensor)
        self.assertEqual(json.loads(result[8:8+size])["x"], header["x"])


class WireAndQualityTests(unittest.TestCase):
    def setUp(self):
        self.identity = {"foundation_hash": "a" * 64, "adapter_hash": "b" * 64, "tokenizer_hash": "c" * 64}
        self.reply = {**self.identity, "results": [{"score": 0x3f800000}]}

    def test_real_finite_bit_contract(self):
        self.assertEqual(check_reply(self.reply, 1, self.identity), [1065353216])
        self.assertEqual(score_value(1065353216), 1.0)

    def test_all_identity_fields_and_literal_prefix_fail(self):
        for field in self.identity:
            changed = {**self.reply, field: "f" * 64}
            with self.assertRaisesRegex(ValueError, "literal"):
                check_reply(changed, 1, self.identity)
        with self.assertRaises(ValueError):
            check_reply({**self.reply, "adapter_hash": "0x" + self.identity["adapter_hash"]}, 1, self.identity)

    def test_nonfinite_invalid_or_conflicting_score_fail(self):
        for value in (True, -1, 1 << 32, 0x7f800000, 0x7fc00000):
            with self.assertRaises(ValueError):
                score_value(value)
        for results in ([], [{"score": 1, "error": "x"}], [{"error": "x"}], [{"score": "1"}]):
            with self.assertRaises(ValueError):
                check_reply({**self.identity, "results": results}, 1, self.identity)

    def reports(self):
        trained = {f: {"n_eval": 128, "pairwise_accuracy": 0.85,
                       "accuracy_by_label": {"Human": 0.8, "AI-generated": 0.9}} for f in FORMATS}
        untrained = {f: {"pairwise_accuracy": 0.5} for f in FORMATS}
        shortcuts = {f: {"length": 0.5, "overlap": 0.5, "answer_only": 0.5} for f in FORMATS}
        return trained, untrained, shortcuts

    def test_good_fixture_quality_passes(self):
        self.assertEqual(quality(*self.reports())["verdict"], "PASS")

    def test_each_quality_gate_has_failure_control(self):
        for mutation in ("overall", "label", "format", "untrained", "shortcut", "count"):
            trained, untrained, shortcuts = self.reports()
            if mutation == "overall": trained["json"]["pairwise_accuracy"] = 0.79
            if mutation == "label": trained["prose"]["accuracy_by_label"]["Human"] = 0.69
            if mutation == "format": trained.pop("bare_label")
            if mutation == "untrained": untrained["prose"]["pairwise_accuracy"] = 0.83
            if mutation == "shortcut": shortcuts["json"]["answer_only"] = 0.8
            if mutation == "count": trained["json"]["n_eval"] = 64
            self.assertEqual(quality(trained, untrained, shortcuts)["verdict"], "FAIL", mutation)


if __name__ == "__main__":
    unittest.main()
