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
                    identity, pair, quality, remote_only, render, sha, audit_excerpts, validation_gate, validation_selection_key)
from checkpoint_contract import check_checkpoint
from development_contract import validate_locked, training_pairs, repetition_gate, layout_training_pairs
from window_contract import validate_plan, validate_augmented, expanded_training_pairs
from training_weights import weight_policy
from preserve import stamp
from reference_render import render as primary_render
from source import parse_article
from static_check import validate_checkout
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
    def test_further_source_windows_are_disjoint_training_annotations(self):
        source_raw = (HERE / "sources.json").read_bytes()
        source = json.loads(source_raw)
        previous = json.loads((HERE / "train-window-plan.json").read_text())
        additional = json.loads((HERE / "additional-train-window-plan.json").read_text())
        validate_plan(additional, source, sha(source_raw))
        self.assertEqual(len(additional["windows"]), 125)
        previous_by_source = {v["source_id"]: v for v in previous["windows"]}
        for view in additional["windows"]:
            old = previous_by_source.get(view["source_id"])
            if old:
                self.assertGreaterEqual(view["prefix_span"][0], old["human_span"][1])
            self.assertEqual(view["split"], "train")
            self.assertEqual(view["human_word_range"], [84, 108])

    def test_larger_context_corpus_cannot_relax_reuse_or_grouping_gates(self):
        # Synthetic values below are isolated structural unit fixtures only.
        source = json.loads((HERE / "sources.json").read_text())
        previous = json.loads((HERE / "train-window-plan.json").read_text())
        additional = json.loads((HERE / "additional-train-window-plan.json").read_text())
        producer = json.loads((HERE / "recovery/locked-augmentation-producer.json").read_text())
        old_raw = (HERE / "recovery/locked-augmentation.jsonl").read_bytes()
        old = validate_augmented(old_raw, previous, source, producer)
        original = [json.loads(s) for s in (HERE / "recovery/locked-development.jsonl").read_text().split("\n") if s]
        fixtures = []
        for view in additional["windows"][:84]:
            human, generated, _ = audit_excerpts(view["human_source_window"],
                "Further structural fixture " + view["view_id"][:8] + " " + " ".join("isolated" + str(i) for i in range(30)))
            base = {"source_id": view["source_id"], "group_id": view["group_id"], "view_id": view["view_id"],
                    "split": "train", "year": view["year"], "source_sha256": view["source_sha256"]}
            fixtures += [{**base, "origin": "human_control", "text": human},
                         {**base, "origin": "synthetic", "text": generated, "generator": "gpt2",
                          "generator_weight_sha256": source["generators"]["gpt2"]["weight_sha256"], "seed": view["seed"]}]
        raw = "".join(json.dumps(r) + "\n" for r in fixtures).encode()
        fixture_report = {"output_sha256": sha(raw), "source_manifest_sha256": additional["source_manifest_sha256"],
                          "generator": source["generators"]["gpt2"], "valid_views": 84}
        new = validate_augmented(raw, additional, source, fixture_report)
        pairs, info = expanded_training_pairs(original, old + new, bare_group_count=6)
        self.assertEqual(len(pairs), 672)
        self.assertEqual(info["bare_pairs"], 12)
        self.assertEqual(info["independent_article_groups"], 128)
        self.assertLessEqual(info["worst_case_internal_negative_leak"], 0.10)
        self.assertFalse(info["gate_thresholds_changed"])
        splits = fixture_splits()
        splits["train"] = pairs
        check_splits(splits)
        with self.assertRaises(ValueError):
            expanded_training_pairs(original, old, bare_group_count=6)
    def test_locked_actual_augmentation_and_symmetric_train_only_weight_policy(self):
        source = json.loads((HERE / "sources.json").read_text())
        plan = json.loads((HERE / "train-window-plan.json").read_text())
        producer = json.loads((HERE / "recovery/locked-augmentation-producer.json").read_text())
        raw = (HERE / "recovery/locked-augmentation.jsonl").read_bytes()
        original = [json.loads(s) for s in (HERE / "recovery/locked-development.jsonl").read_text().split("\n") if s]
        added = validate_augmented(raw, plan, source, producer)
        rows, expansion = expanded_training_pairs(original, added)
        weights, policy = weight_policy(rows, 16)
        self.assertEqual(len(rows), 504)
        self.assertEqual(len(weights), 504)
        self.assertEqual(expansion["bare_pairs"], 8)
        self.assertEqual(policy["total_training_weight"], 624)
        self.assertEqual(policy["weight_by_origin"], {"Human": 312.0, "AI-generated": 312.0})
        _, stronger = weight_policy(rows, 32)
        self.assertEqual(stronger["total_training_weight"], 752.0)
        self.assertEqual(stronger["weight_by_origin"], {"Human": 376.0, "AI-generated": 376.0})
        self.assertFalse(policy["acceptance_thresholds_changed"])
        self.assertEqual({r["split"] for r in rows}, {"train"})
        invalid = copy.deepcopy(rows)
        invalid[0]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "training pairs only"):
            weight_policy(invalid, 16)
        with self.assertRaises(ValueError):
            weight_policy(rows, 100)
        poisoned = copy.deepcopy(producer)
        poisoned["output_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            validate_augmented(raw, plan, source, poisoned)

    def test_source_window_plan_is_train_only_attributed_and_disjoint(self):
        source_raw = (HERE / "sources.json").read_bytes()
        source = json.loads(source_raw)
        plan = json.loads((HERE / "train-window-plan.json").read_text())
        validate_plan(plan, source, sha(source_raw))
        self.assertEqual(len(plan["windows"]), 127)
        bad = copy.deepcopy(plan)
        bad["windows"][0]["split"] = "validation"
        with self.assertRaises(ValueError):
            validate_plan(bad, source, sha(source_raw))
        bad = copy.deepcopy(plan)
        bad["windows"][0]["human_span"][0] = 0
        with self.assertRaises(ValueError):
            validate_plan(bad, source, sha(source_raw))

    def test_expanded_pair_structure_preserves_grouping_and_hard_reuse_bound(self):
        # Constructed structural unit fixtures only. These are never written to
        # the generator output or used as new training/audit data.
        source = json.loads((HERE / "sources.json").read_text())
        plan = json.loads((HERE / "train-window-plan.json").read_text())
        original = [json.loads(s) for s in (HERE / "recovery/locked-development.jsonl").read_text().split("\n") if s]
        fake = []
        for view in plan["windows"][:80]:
            human, generated, _ = audit_excerpts(view["human_source_window"],
                "Constructed unit fixture " + view["view_id"][:8] + " " + " ".join("testword" + str(i) for i in range(30)))
            base = {"source_id": view["source_id"], "group_id": view["group_id"], "view_id": view["view_id"],
                    "split": "train", "year": view["year"], "source_sha256": view["source_sha256"]}
            fake += [{**base, "origin": "human_control", "text": human},
                     {**base, "origin": "synthetic", "text": generated, "generator": "gpt2",
                      "generator_weight_sha256": source["generators"]["gpt2"]["weight_sha256"], "seed": view["seed"]}]
        raw = "".join(json.dumps(r) + "\n" for r in fake).encode()
        report = {"output_sha256": sha(raw), "source_manifest_sha256": plan["source_manifest_sha256"],
                  "generator": source["generators"]["gpt2"], "valid_views": 80}
        validated = validate_augmented(raw, plan, source, report)
        pairs, counters = expanded_training_pairs(original, validated)
        self.assertEqual(len(pairs), 416)
        self.assertEqual(counters["independent_article_groups"], 128)
        self.assertEqual(counters["bare_pairs"], 8)
        self.assertLessEqual(counters["worst_case_internal_negative_leak"], 0.10)
        self.assertFalse(counters["gate_thresholds_changed"])
        split_rows = fixture_splits()
        split_rows["train"] = pairs
        check_splits(split_rows)
        with self.assertRaises(ValueError):
            expanded_training_pairs(original, validated[:100])

    def test_weighting_cannot_introduce_origin_asymmetry(self):
        rows = fixture_splits()["train"]
        _, policy = weight_policy(rows, 16)
        self.assertEqual(policy["mean_training_weight"], 1.0)
        rows[0]["style"] = "bare_label"
        with self.assertRaisesRegex(ValueError, "symmetric"):
            weight_policy(rows, 16)
    def test_locked_development_primary_path_and_roles(self):
        raw = (HERE / "recovery/locked-development.jsonl").read_bytes()
        producer = json.loads((HERE / "recovery/locked-development-producer.json").read_text())
        lineage = json.loads((HERE / "recovery/locked-development-lineage.json").read_text())
        source = json.loads((HERE / "sources.json").read_text())
        self.assertEqual(sha((HERE / "sources.json").read_bytes()), lineage["source_manifest_sha256"])
        records = validate_locked(raw, source, producer, lineage)
        self.assertEqual(len(records), 320)
        self.assertEqual(sum(r["split"] == "train" for r in records), 256)
        self.assertEqual(sum(r["split"] == "validation" for r in records), 64)
        self.assertTrue(all(r["split"] != "test" for r in records))
        with self.assertRaisesRegex(ValueError, "byte binding"):
            validate_locked(raw + b"x", source, producer, lineage)
        bad = copy.deepcopy(records)
        bad[-1]["split"] = "train"
        changed_raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in bad).encode()
        changed_lineage = {**lineage, "data_sha256": sha(changed_raw)}
        changed_producer = {**producer, "output_sha256": sha(changed_raw)}
        with self.assertRaisesRegex(ValueError, "source or split role"):
            validate_locked(changed_raw, source, changed_producer, changed_lineage)

    def test_exact_repeated_negative_failure_and_unique_bare_repair(self):
        records = [json.loads(line) for line in (HERE / "recovery/locked-development.jsonl").read_text().split("\n") if line]
        failed = repetition_gate(training_pairs(records, bare_group_count=4))
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["internal_holdout"], 38)
        self.assertAlmostEqual(failed["eval_negative_leak"], 4 / 38)
        fixed_rows = training_pairs(records)
        self.assertEqual(sum(r["style"] == "bare_label" for r in fixed_rows), 2)
        for seed in (42, 0, 1, 7, 123):
            repaired = repetition_gate(fixed_rows, seed=seed)
            self.assertTrue(repaired["passed"])
            self.assertEqual(repaired["unique_rejected_frac"], 1.0)
            self.assertEqual(repaired["eval_negative_leak"], 0.0)
            self.assertEqual(repaired["pairs"], 256)

    def test_fresh_audit_quarantines_old_articles_and_named_authors(self):
        spec = json.loads((HERE / "sources.json").read_text())
        refreshed = spec["audit_refresh"]
        current = spec["source_report"]["articles"]["test"]
        self.assertEqual(len(current), 64)
        self.assertTrue(refreshed["train_validation_unchanged"])
        self.assertFalse({r["source_id"] for r in current}.intersection(refreshed["retired_article_ids"]))
        self.assertFalse({a for r in current for a in r["author_ids"]}.intersection(refreshed["retired_author_ids"]))

    def test_exact_original_checkpoint_and_failure_controls(self):
        raw = (HERE / "recovery/original-checkpoint.safetensors").read_bytes()
        original_sha = "6c34c23e77582cbb14ff2daf579c11fc4bcf9674541e9553ab2df810bbf06636"
        original_source = "a38863c8d1d0643b250bc19c84c0d7c1dfb6fe008b56ab8670f271a90630c567"
        binding = check_checkpoint(raw, original_sha, original_source)
        self.assertFalse(binding["tensor_payload_interpreted"])
        with self.assertRaisesRegex(ValueError, "byte identity"):
            check_checkpoint(raw + b"x", original_sha, original_source)
        with self.assertRaisesRegex(ValueError, "source lineage"):
            check_checkpoint(raw, original_sha, "f" * 64)

    def test_selected_improved_checkpoint_is_exact_and_licensed(self):
        recipe = json.loads((HERE / "recipe.json").read_text())
        raw = (HERE / recipe["checkpoint_file"]).read_bytes()
        checked = check_checkpoint(raw, recipe["checkpoint_sha256"], recipe["parent_source_manifest_sha256"])
        self.assertFalse(checked["tensor_payload_interpreted"])
        self.assertIn(recipe["continuation_run_id"], (44588237548, 44632309433))

    def test_weight16_predecessor_remains_retained_when_followup_is_selected(self):
        recipe = json.loads((HERE / "recipe.json").read_text())
        raw = (HERE / "recovery/weight16-checkpoint.safetensors").read_bytes()
        checked = check_checkpoint(raw, "39c3b89efa9da2c61ab841d82a7ceb31ba2da1acafbf6d2710163a987a073efe",
                                   recipe["source_manifest_sha256"])
        self.assertFalse(checked["tensor_payload_interpreted"])
        actual = json.loads((HERE / "recovery/weight16-validation.json").read_text())
        self.assertEqual(actual["formats"]["bare_label"]["pairwise_accuracy"], 0.7656)
        self.assertFalse(actual["gate"]["passed"])

    def test_train_only_layout_diversity_preserves_pairs_and_byte_contract(self):
        records = [json.loads(line) for line in (HERE / "recovery/locked-development.jsonl").read_text().split("\n") if line]
        old, changed = training_pairs(records), layout_training_pairs(records)
        self.assertEqual(len(old), len(changed))
        self.assertEqual({r["layout"] for r in changed if r["style"] != "bare_label"}, {"verdict_first", "verdict_last"})
        for before, after in zip(old, changed):
            for key in ("prompt", "source_id", "group_id", "label", "origin", "split"):
                self.assertEqual(before[key], after[key])
            self.assertEqual(after["split"], "train")
            self.assertNotEqual(after["chosen"], after["rejected"])
            if after["style"] == "json":
                self.assertEqual(json.loads(before["chosen"]), json.loads(after["chosen"]))
                self.assertEqual(json.loads(before["rejected"]), json.loads(after["rejected"]))
        self.assertTrue(repetition_gate(changed)["passed"])
        self.assertEqual(repetition_gate(changed)["unique_rejected_frac"], 1.0)

    def test_symmetric_byte_fit_does_not_relax_limits_or_identity(self):
        human = " ".join("humanword" + str(i) for i in range(24))
        generated = "\n\n".join("generated" + str(i) for i in range(24))
        h, g, view = audit_excerpts(human, generated)
        self.assertEqual(len(h.split()), len(g.split()))
        self.assertLessEqual(len(h.encode()), 256)
        self.assertLessEqual(len(g.encode()), 256)
        self.assertGreaterEqual(view["word_count"], 16)
        with self.assertRaisesRegex(ValueError, "identical"):
            audit_excerpts(human, human)
        with self.assertRaisesRegex(ValueError, "unchanged request ceiling"):
            audit_excerpts(human, " ".join("x" * 256 for _ in range(24)))

    def test_actual_standalone_checkout_contract_and_failure_control(self):
        config = json.loads((HERE / "workflow.json").read_text())
        validate_checkout(config)
        changed = copy.deepcopy(config)
        changed["jobs"]["verify"]["steps"][0]["with"]["sparse-checkout"] = "/verification/owners/ai_text_detection/"
        with self.assertRaisesRegex(ValueError, "exact source and workflow"):
            validate_checkout(changed)

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
            from train_formats import fit
            with self.assertRaises(RuntimeError):
                fit(*([None] * 8))
            import augment_training
            with self.assertRaises(RuntimeError):
                augment_training.main()

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

    def test_validation_seals_audit_on_any_failed_format_or_label(self):
        metrics = {f: {"n_eval": 64, "pairwise_accuracy": 0.85,
                      "accuracy_by_label": {"Human": 0.8, "AI-generated": 0.9}} for f in FORMATS}
        baseline = {f: {"length": 0.5, "overlap": 0.5, "answer_only": 0.5} for f in FORMATS}
        self.assertTrue(validation_gate(metrics, baseline)["passed"])
        metrics["bare_label"]["pairwise_accuracy"] = 0.5469
        self.assertFalse(validation_gate(metrics, baseline)["passed"])
        metrics["bare_label"]["pairwise_accuracy"] = 0.85
        metrics["prose"]["accuracy_by_label"]["Human"] = 0.69
        self.assertFalse(validation_gate(metrics, baseline)["passed"])

    def test_epoch_selection_cannot_hide_a_failed_format_with_good_mean_loss(self):
        base = {f: {"n_eval": 64, "pairwise_accuracy": 0.85, "val_loss": 0.8,
                    "accuracy_by_label": {"Human": 0.8, "AI-generated": 0.9}} for f in FORMATS}
        baseline = {f: {"length": 0.5, "overlap": 0.5, "answer_only": 0.5} for f in FORMATS}
        failed = copy.deepcopy(base)
        for m in failed.values():
            m["val_loss"] = 0.01
        failed["bare_label"]["pairwise_accuracy"] = 0.7188
        failed["bare_label"]["accuracy_by_label"]["Human"] = 0.6875
        self.assertGreater(validation_selection_key(base, baseline), validation_selection_key(failed, baseline))
        broken = copy.deepcopy(base)
        broken["json"]["n_eval"] = 63
        with self.assertRaises(ValueError):
            validation_selection_key(broken, baseline)


if __name__ == "__main__":
    unittest.main()
