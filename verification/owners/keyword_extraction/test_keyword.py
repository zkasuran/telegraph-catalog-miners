# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source, lexical, group and literal wire failure controls. No model operations."""
import copy
import json
import os
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from assemble import validate_reviews
from audit_remote import (
    FORMATS,
    identity_controls,
    metrics,
    registration_report,
    verdict,
    views,
)
from gates import check_pilot_counts, check_splits, fit_baselines, shortcut_report
from prepare_data import answer, group_sources, prompt, whole_annotation
from preserve import stamp
from ready_contract import ENGINE, FOUNDATION, TOKENIZER
from remote import remote_only, verify_sources
from scorer import render, score, terms
from source import blob, contains, parse, verify_tree
from wire import check_response


def fixture(doi="10.7554/eLife.fixture", author="Test Author", group="fixture"):
    source = {"doi": doi, "authors": [author], "text": "Test-only synthetic fixture, not scientific data",
              "licence": "CC0-1.0", "grant": "Test fixture only", "url": "https://example.com/fixture",
              "abstract": "fixture"}
    return {"prompt": "fixture alpha beta", "chosen": "alpha", "rejected": "beta", "source": source,
            "group_id": group, "label_kind": "preference_pair", "review_origin": "AI-assisted",
            "centrality_reason": "Synthetic test of validation mechanics only, not a preference annotation"}


class KeywordTests(unittest.TestCase):
    def test_review_identity_counts_and_origins(self):
        # Synthetic decisions exercise validation only, never scientific labels.
        decision = {"proposal_id": "test-only", "status": "approve", "review_origin": "AI-assisted",
                    "choice_a": "term_a", "choice_b": "term_b", "reason_a": "Synthetic A", "reason_b": "Synthetic B"}
        review = {"schema_version": 1, "reviews": [decision], "approved_contrasts": 1, "approved_pairs": 2}
        proposals = {"test-only": {}}
        self.assertEqual(validate_reviews(review, proposals), [decision])
        for mutation, message in (
            ({"schema_version": True}, "unsupported review schema"),
            ({"reviews": [decision, decision]}, "duplicate or conflicting"),
            ({"reviews": [decision, {"proposal_id": "test-only", "status": "reject", "reason": "Synthetic rejection"}]}, "duplicate or conflicting"),
            ({"approved_pairs": 4}, "count mismatch"),
            ({"approved_contrasts": True}, "count mismatch"),
            ({"reviews": [{**decision, "proposal_id": "unknown"}]}, "unknown review proposal"),
            ({"reviews": [{**decision, "review_origin": "human-reviewed"}]}, "origin required"),
            ({"reviews": [{**decision, "reason_a": "   "}]}, "reasoning required"),
            ({"reviews": [{**decision, "status": "unreviewed"}]}, "unknown review decision"),
            ({"reviews": [{**decision, "choice_b": "term_a"}]}, "opposite centrality"),
            ({"reviews": [{**decision, "choice_a": []}]}, "opposite centrality"),
        ):
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, message):
                validate_reviews({**review, **mutation}, proposals)

    def test_primary_keyword_smoke(self):
        passage = "CRISPR gene editing\nCRISPR gene editing changes DNA in cells."
        good = score(passage, '["CRISPR","gene editing"]')
        bad = score(passage, '["continental drift","astrophysics"]')
        self.assertGreater(good["score_milli"], bad["score_milli"])
        self.assertTrue(all(t["literal"] for t in good["terms"]))

    def test_no_morphological_substring_pass(self):
        self.assertFalse(contains("dopaminergic circuits and navigational bias", "dopamine"))
        self.assertFalse(contains("dopaminergic circuits and navigational bias", "navigation"))
        self.assertFalse(contains("an antibody", "body"))
        self.assertTrue(contains("The Dopamine circuit.", "dopamine"))

    def test_literal_unicode_and_comma_term(self):
        self.assertEqual(terms("PI(4,5)P2, cell wall"), ["PI(4,5)P2", "cell wall"])
        with self.assertRaisesRegex(ValueError, "ambiguous numeric comma"):
            terms("β-1,6-Glucan")
        self.assertEqual(terms('["PI(4,5)P2","β-1,6-Glucan"]'), ["PI(4,5)P2", "β-1,6-Glucan"])
        self.assertTrue(contains("PI(4,5)P2 binds Dia", "PI(4,5)P2"))

    def test_duplicate_penalty_and_formats(self):
        p = "Gene editing\nGene editing alters DNA."
        self.assertEqual(score(p, '["gene editing"]')["score_milli"], score(p, '{"keywords":["gene editing"]}')["score_milli"])
        self.assertLess(score(p, '["gene editing","GENE EDITING"]')["score_milli"], score(p, '["gene editing"]')["score_milli"])

    def test_hostile_answer_bounds(self):
        for value in ("", "[]", '{"score":1}', "[1]", "broken(", "x)", "; ", "x" * 4097, "[" * 4097):
            with self.subTest(value=value[:20]), self.assertRaises(ValueError):
                terms(value)
        with self.assertRaises(ValueError):
            score("", "word")

    def test_random_parser_bounds(self):
        rng = random.Random(619)
        for _ in range(200):
            value = "".join(chr(rng.randrange(32, 127)) for _ in range(rng.randrange(64)))
            try:
                parsed = terms(value)
                self.assertTrue(1 <= len(parsed) <= 32)
            except ValueError:
                pass

    def test_whole_source_smoke(self):
        root = Path(__file__).parent / "fixtures"
        entry = json.loads((root / "source.json").read_text())
        raw = (root / entry["path"]).read_bytes()
        row = parse(raw, entry)
        self.assertEqual(row["doi"], "10.7554/eLife.27706")
        self.assertIn("Angiogenesis", row["author_keywords"])
        self.assertIn("source are credited", row["grant"])
        self.assertGreater(len(row["abstract"]), 500)

    def test_tampered_blob_and_entities_refused(self):
        with self.assertRaisesRegex(ValueError, "blob mismatch"):
            parse(b"<article/>", {"sha": "0" * 40})
        entity = b'<!ENTITY x "danger"><article/>'
        with self.assertRaisesRegex(ValueError, "entity declaration"):
            parse(entity, {"sha": blob(entity)})

    def test_wrong_source_licence_refused(self):
        raw = b'<article article-type="research-article"><front><article-meta><permissions><license xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="https://example.com/no-grant">Unknown</license></permissions></article-meta></front></article>'
        with self.assertRaisesRegex(ValueError, "unreviewed article grant"):
            parse(raw, {"sha": blob(raw)})

    def test_inventory_tampering_refused(self):
        with self.assertRaisesRegex(ValueError, "tree hash mismatch"):
            verify_tree({"sha": "0" * 40, "truncated": False, "tree": [{"path": "file.xml", "mode": "100644", "type": "blob", "sha": "1" * 40}]})
        with self.assertRaisesRegex(ValueError, "truncated"):
            verify_tree({"truncated": True, "tree": []})

    def test_author_union_before_filter(self):
        a, b, c = (fixture(doi=str(i), author=str(i), group=str(i))["source"] for i in range(3))
        a["authors"], b["authors"], c["authors"] = ["A"], ["A", "B"], ["B"]
        for i, row in enumerate((a, b, c)):
            row["text"], row["abstract"] = str(i), str(i)
            row["topics"] = ["Neuroscience"] if i == 2 else ["Cell Biology"]
        groups = group_sources([a, b, c])
        self.assertEqual(len(groups), 1)
        self.assertEqual(a["group_id"], c["group_id"])
        self.assertIn("Neuroscience", a["component_topics"])

    def test_unreviewed_and_overlap_refused(self):
        splits = {s: [fixture(doi=s, author=s, group=s)] for s in ("train", "validation", "test")}
        for s, rows in splits.items():
            rows[0]["source"]["text"] = s
            rows[0]["chosen"], rows[0]["rejected"] = s + " A", s + " B"
        self.assertTrue(check_splits(splits))
        bad = copy.deepcopy(splits)
        bad["test"][0]["review_origin"] = "unreviewed"
        with self.assertRaisesRegex(ValueError, "unreviewed"):
            check_splits(bad)
        bad = copy.deepcopy(splits)
        bad["test"][0]["source"]["authors"] = [" TRAIN "]
        with self.assertRaisesRegex(ValueError, "author overlap"):
            check_splits(bad)
        with self.assertRaisesRegex(ValueError, "empty"):
            check_splits({"train": [], "validation": [], "test": []})

    def test_frozen_integer_baseline(self):
        row = fixture()
        inverse = {**row, "chosen": row["rejected"], "rejected": row["chosen"]}
        frozen = fit_baselines([row, inverse])
        original = json.dumps(frozen, sort_keys=True)
        self.assertEqual(shortcut_report([row, inverse], frozen), {"length": 0.5, "overlap": 0.5, "answer_only": 0.5})
        self.assertEqual(original, json.dumps(frozen, sort_keys=True))

    def test_literal_wire_prefix_control(self):
        identity = {"foundation_hash": "aabb", "adapter_hash": "ccdd", "tokenizer_hash": "eeff"}
        reply = {**identity, "results": [{"score": 1065353216}]}
        self.assertEqual(check_response(reply, identity, 1), [1065353216])
        reply["adapter_hash"] = "CCDD"
        self.assertEqual(check_response(reply, identity, 1), [1065353216])
        reply["adapter_hash"] = "0xCCDD"
        with self.assertRaisesRegex(ValueError, "literal runtime identity"):
            check_response(reply, identity, 1)

    def test_nonfinite_and_missing_wire_result(self):
        identity = {"foundation_hash": "aa", "adapter_hash": "bb", "tokenizer_hash": "cc"}
        for item in ({"score": 2139095040}, {"score": True}, {"error": "bad tokens"}, {}, {"score": -1}):
            with self.subTest(item=item), self.assertRaises(ValueError):
                check_response({**identity, "results": [item]}, identity, 1)

    def test_literal_render_not_normalized(self):
        p, a = "  α\nTask", '{"keywords":["β"]}  '
        self.assertEqual(render(p, a), "### Task:\n  α\nTask\n\n### Answer:\n{\"keywords\":[\"β\"]}  ")

    def test_remote_guard_refuses_laptop(self):
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "restricted"):
            remote_only()

    def test_config_validates(self):
        cfg = json.loads((Path(__file__).parent / "config.json").read_text())
        self.assertEqual(cfg["family"], "KEYWORD_EXTRACTION")
        self.assertEqual(cfg["max_tokens"], 384)
        self.assertEqual(cfg["formats"], ["plain", "array", "object"])
        self.assertLessEqual(cfg["timeout_seconds"], 6900)
        self.assertLessEqual(cfg["max_rss_bytes"], 10 << 30)
        self.assertEqual(cfg["maximum_data_shortcut"], 0.65)

    def test_complete_annotation_wire_is_shared_without_cropping(self):
        row = {"author_keywords": ["α", "PI(4,5)P2", "cell wall"]}
        value = whole_annotation(row)
        self.assertEqual(value, '{"keywords":["α","PI(4,5)P2","cell wall"]}')
        self.assertEqual(json.loads(value)["keywords"], row["author_keywords"])

    def test_pilot_counts_cannot_accept_tiny_splits(self):
        with self.assertRaisesRegex(ValueError, "insufficient reviewed pairs"):
            check_pilot_counts({s: [fixture()] for s in ("train", "validation", "test")})

    def test_repeated_source_claim_is_reverified(self):
        # Only the inventory binding is mocked to isolate repeated row handling.
        # Source bytes and the article-specific grant are the real pinned fixture.
        directory = Path(__file__).parent / "fixtures"
        entry = json.loads((directory / "source.json").read_text())
        raw = (directory / entry["path"]).read_bytes()
        source = parse(raw, entry)
        row = {"source": source, "prompt": prompt(source), "chosen_term": "SOX4", "rejected_term": "Angiogenesis",
               "chosen": answer("SOX4", "array"), "rejected": answer("Angiogenesis", "array")}
        forged = copy.deepcopy(row)
        forged["source"]["grant"] = "Fabricated grant"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "inventory.json").write_text(json.dumps({"sha": "4ee79e98a7dbdcd3dd040be4b836e0843520c1fd", "tree": [entry]}))
            with mock.patch("remote.verify_tree"), mock.patch("remote.source_fetch", return_value=raw) as fetcher:
                with self.assertRaisesRegex(ValueError, "grants differ"):
                    verify_sources(root, {"train": [row, forged]})
                self.assertEqual(fetcher.call_count, 1)

    def test_original_prompt_cannot_be_cropped(self):
        directory = Path(__file__).parent / "fixtures"
        entry = json.loads((directory / "source.json").read_text())
        raw = (directory / entry["path"]).read_bytes()
        source = parse(raw, entry)
        row = {"source": source, "prompt": "Cropped source", "chosen_term": "SOX4", "rejected_term": "Angiogenesis",
               "chosen": answer("SOX4", "array"), "rejected": answer("Angiogenesis", "array")}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "inventory.json").write_text(json.dumps({"sha": "4ee79e98a7dbdcd3dd040be4b836e0843520c1fd", "tree": [entry]}))
            with mock.patch("remote.verify_tree"), mock.patch("remote.source_fetch", return_value=raw), self.assertRaisesRegex(ValueError, "complete source rendering"):
                verify_sources(root, {"train": [row]})

    def test_audit_formats_and_group_weighting(self):
        rows = [{**fixture(group="A"), "chosen_term": "alpha", "rejected_term": "beta"},
                {**fixture(group="A"), "chosen_term": "gamma", "rejected_term": "delta"},
                {**fixture(group="B"), "chosen_term": "epsilon", "rejected_term": "zeta"}]
        self.assertEqual(set(views(rows)), set(FORMATS))
        m = metrics(rows, [1065353216, 0, 1065353216, 0, 0, 1065353216])
        self.assertEqual(m["pairwise_accuracy"], 2 / 3)
        self.assertEqual(m["group_accuracy"], 0.5)
        self.assertEqual(m["labels"]["preference_pair"]["accuracy"], 2 / 3)
        self.assertEqual(metrics(rows[:1], [0, 0])["correct"], 0)
        with self.assertRaises(ValueError):
            metrics(rows, [0])

    def test_audit_rejects_shortcut_and_initialization_failure(self):
        config = json.loads((Path(__file__).parent / "config.json").read_text())
        rows = [fixture(group=str(i // 2)) for i in range(64)]
        good = metrics(rows, [1065353216, 0] * 64)
        bad = metrics(rows, [0, 1065353216] * 64)
        trained = {name: good for name in FORMATS}
        initial = {name: bad for name in FORMATS}
        self.assertEqual(verdict(trained, initial, {"length": 0.5}, config)["verdict"], "PASS")
        self.assertEqual(verdict(trained, trained, {"length": 0.5}, config)["verdict"], "FAIL")
        self.assertEqual(verdict(trained, initial, {"fixed_lexical": 1.0}, config)["verdict"], "FAIL")
        tiny = {name: metrics(rows[:1], [1065353216, 0]) for name in FORMATS}
        self.assertEqual(verdict(tiny, tiny, {"length": 0.5}, config)["verdict"], "FAIL")
        self.assertEqual(verdict(trained, initial, {"answer_only": 0.66}, config)["verdict"], "FAIL")

    def test_actual_identity_control_contract_with_synthetic_exchange(self):
        # Fake server returns its own identity like the primary handler. No neural work.
        identity = {"foundation_hash": "aa", "adapter_hash": "bb", "tokenizer_hash": "cc", "input_root": "dd"}
        def exchange(batch, requested):
            return {**identity, "results": [{"score": 1065353216} for _ in batch]}
        controls = identity_controls(exchange, [[50281, 1, 50282]], identity, [1065353216])
        self.assertEqual(len(controls), 4)
        def echo(batch, requested):
            return {**requested, "results": [{"score": 1065353216} for _ in batch]}
        with self.assertRaisesRegex(ValueError, "literal runtime identity"):
            identity_controls(echo, [[50281, 1, 50282]], identity, [1065353216])

    def test_registration_emitter_refuses_unmeasured_or_failed_controls(self):
        # Complete synthetic schema fixture only. It is never written as a ready artifact.
        rows = [fixture(group=str(i // 2)) for i in range(64)]
        good = metrics(rows, [1065353216, 0] * 64)
        initial = metrics(rows, [0, 1065353216] * 64)
        audit = {"verdict": "PASS", "candidate": {"package_sha256": "b" * 64}, "audit_run_id": "1", "audit_commit": "a" * 40,
                 "formats": {"candidate": {name: good for name in FORMATS}, "initial": {name: initial for name in FORMATS}},
                 "format_shortcuts": {name: {"synthetic_only": 0.5} for name in FORMATS},
                 "normalized_overlap": 0, "fixed_vector_bits": [1065353216] * 8, "executor_sha256": "f" * 64,
                 "runtime": {"reference_engine_passed": True, "repeat_bits_identical": True, "token_control": {"negative_tokens_refused": True}},
                 "failure_controls": {k: {"supplied": "0xaa", "returned": "aa", "validator_refused": True, "score_bits_unchanged": True}
                                      for k in ("foundation_hash", "adapter_hash", "tokenizer_hash", "input_root")}}
        manifest = {"sha256": "b" * 64, "runtime_adapter_hash": "c" * 64, "manifest": {"verification": {"stage1": True}}}
        assets = {"foundation_lora_fp32.onnx": FOUNDATION, "foundation_engine.wasm": ENGINE, "tokenizer.json": TOKENIZER}
        vectors = [[50281, 1, 50282]] * 8
        verification = {"verified_originals": 64}
        report = registration_report(audit, manifest, "d" * 64, verification, vectors, assets)
        self.assertEqual(report["quality"]["formats"]["plain"]["label_counts"], {"preference_pair": 64})
        for key, value in (("verdict", "FAIL"), ("normalized_overlap", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                registration_report({**audit, key: value}, manifest, "d" * 64, verification, vectors, assets)
        bad = copy.deepcopy(audit)
        bad["failure_controls"]["adapter_hash"]["validator_refused"] = False
        with self.assertRaisesRegex(ValueError, "identity controls"):
            registration_report(bad, manifest, "d" * 64, verification, vectors, assets)
        bad = copy.deepcopy(audit)
        bad["runtime"]["token_control"]["negative_tokens_refused"] = False
        with self.assertRaisesRegex(ValueError, "negative token refusal"):
            registration_report(bad, manifest, "d" * 64, verification, vectors, assets)
        with self.assertRaises(ValueError):
            registration_report(audit, manifest, "d" * 64, verification, vectors, {**assets, "foundation_engine.wasm": "0" * 64})
        with self.assertRaises(ValueError):
            registration_report(audit, manifest, "d" * 64, verification, vectors[:7], assets)
        with self.assertRaisesRegex(ValueError, "package identity"):
            registration_report(audit, {**manifest, "sha256": "e" * 64}, "d" * 64, verification, vectors, assets)

    def test_input_root_prefix_and_missing_echo_refused(self):
        identity = {"foundation_hash": "aa", "adapter_hash": "bb", "tokenizer_hash": "cc", "input_root": "abcd"}
        reply = {**identity, "results": [{"score": 1065353216}]}
        self.assertEqual(check_response(reply, identity, 1), [1065353216])
        reply["input_root"] = "0xabcd"
        with self.assertRaisesRegex(ValueError, "input root"):
            check_response(reply, identity, 1)
        del reply["input_root"]
        with self.assertRaisesRegex(ValueError, "input root"):
            check_response(reply, identity, 1)

    def test_standalone_support_does_not_load_neural_modules(self):
        import native
        self.assertEqual(native.SOURCES["kit"]["revision"], "1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd")
        self.assertFalse({"torch", "transformers", "peft", "onnxruntime"}.intersection(sys.modules))
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "restricted"):
            native.prepare(Path("unused-local-model-directory"))

    def test_remote_guard_distinguishes_public_workflow_context(self):
        with mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "zkasuran/telegraph-catalog-miners"}, clear=True):
            self.assertIsNone(remote_only())
        with mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "wrong/owner"}, clear=True), self.assertRaises(RuntimeError):
            remote_only()
        with mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "zkasuran/telegraph-catalog-miners", "OWNER_SOURCE_ONLY": "1"}, clear=True), self.assertRaises(RuntimeError):
            remote_only()

    def test_checkpoint_stamp_is_byte_preserving(self):
        header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        original = struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0)
        stamped = stamp(original, {"notice": "Original grants retained; synthetic parser fixture only"})
        size = struct.unpack("<Q", stamped[:8])[0]
        self.assertEqual(stamped[8 + size:], original[8 + len(header):])
        self.assertIn("x-terms-notice", json.loads(stamped[8:8 + size])["__metadata__"])


if __name__ == "__main__":
    unittest.main()
