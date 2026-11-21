# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Artificial fixtures test parsers and gates only. They are never training data."""
import copy
import json
import math
import os
import random
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import checkpoint
import baselines
import data
import scorer
import source
import wire
import overlap


def record(ident, title, doi=None):
    oid = "https://openalex.org/W" + ident
    return {"id": oid, "title": title, "doi": doi, "ids": {"openalex": oid, "pmid": "https://pubmed.ncbi.nlm.nih.gov/" + ident}, "publication_year": 2020}


class Contracts(unittest.TestCase):
    def test_literal_render(self):
        self.assertEqual(data.render(" Q\n", " A "), "### Task:\n Q\n\n\n### Answer:\n A ")

    def test_hashes_are_not_normalized(self):
        request = wire.request([[1, 2]], "a" * 64, "b" * 64)
        response = {k: v for k, v in request.items() if k != "batch"}
        response["results"] = [{"score": 0x3f800000}]
        self.assertEqual(wire.response(json.dumps(response).encode(), request), [0x3f800000])
        for key in ("foundation_hash", "tokenizer_hash", "adapter_hash", "input_root"):
            for bad in ("0x" + response[key], response[key].upper(), " " + response[key], "c" * 64):
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    wire.response(json.dumps({**response, key: bad}).encode(), request)

    def test_request_refuses_bad_tokens_and_single_item_standin(self):
        for batch in ([], [[True]], [[-1]], [[50368]], [[1] * 257], [[1.0]], "tokens", [[1]] * 101):
            with self.assertRaises(ValueError):
                wire.request(batch, "a" * 64, "b" * 64)
        for value in (None, "0x" + "a" * 64, "a" * 63):
            with self.assertRaises(ValueError):
                wire.request([[1]], value, "b" * 64)

    def test_finite_score_bits_and_errors(self):
        for bits in (True, -1, 2 ** 32, 1.0, 0x7fc00000, 0x7f800000, 0xff800000):
            with self.assertRaises(ValueError):
                wire.score_value(bits)
        self.assertEqual(wire.score_value(0x3f800000), 1.0)
        self.assertEqual(wire.score_value(0x80000000), 0.0)

    def test_duplicate_and_nonfinite_json(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'x' * (wire.MAX_BODY + 1)):
            with self.assertRaises(ValueError):
                wire.strict_json(raw)

    def test_wire_missing_result_never_passes(self):
        sent = wire.request([[1], [2]], "a" * 64, "b" * 64)
        value = {k: v for k, v in sent.items() if k != "batch"}
        for results in ([], [{"score": 1}], [{"score": 1}, {"error": "invalid token"}], [{"score": 1}, {"score": True}]):
            with self.assertRaises(ValueError):
                wire.response(json.dumps({**value, "results": results}).encode(), sent)

    def test_property_untrusted_wire_bytes(self):
        rng = random.Random(42)
        sent = wire.request([[1]], "a" * 64, "b" * 64)
        for _ in range(100):
            raw = bytes(rng.randrange(256) for _ in range(rng.randrange(128)))
            with self.assertRaises((ValueError, UnicodeError)):
                wire.response(raw, sent)

    def test_complete_metadata_batch(self):
        body = {"meta": {"count": 1}, "results": [record("1", "title")]}
        got, missing, ambiguous = source.validate_batch(body, ["1", "2"])
        self.assertEqual((list(got), missing, ambiguous), (["1"], ["2"], []))
        with self.assertRaises(ValueError):
            source.validate_batch({**body, "meta": {"count": 200}}, ["1"])

    def test_metadata_identity_and_field_failures(self):
        for row in ({**record("1", "title"), "abstract": "forbidden"}, record("2", "other"), {**record("1", "title"), "ids": {}}, None):
            with self.assertRaises(ValueError):
                source.validate_batch({"meta": {"count": 1}, "results": [row]}, ["1"])

    def test_ambiguous_binding_excluded(self):
        got, missing, multi = source.validate_batch({"meta": {"count": 2}, "results": [record("1", "one"), record("1", "one")]}, ["1"])
        self.assertEqual((got, missing, multi), ({}, ["1"], ["1"]))

    def test_transitive_metadata_aliases(self):
        records = {"1": record("1", "Ａ Paper", "https://doi.org/10.1/a"), "2": record("2", "a paper", "https://doi.org/10.1/b"), "3": record("3", "third", "https://doi.org/10.1/B")}
        components = data.entities(records)
        self.assertEqual(components.find("1"), components.find("3"))

    def test_human_annotation_disagreement_excluded(self):
        rows = [{"uid": "a", "pmid": "1", "relevant": ["2", "3"], "partial": [], "irrelevant": ["4"]},
                {"uid": "b", "pmid": "1", "relevant": ["3"], "partial": ["2"], "irrelevant": ["4"]}]
        pool, report = source.consensus(rows)
        self.assertEqual([r["pmid"] for r in pool["1"]["relevant"]], ["3"])
        self.assertEqual(report["conflicting_pairs_excluded"], 1)

    def test_normalization_and_format_original_preservation(self):
        self.assertEqual(data.norm(" Ａ\tB\u0085C "), "a b c")
        original = " Title\nwith  spacing "
        rendered = data.formats(original, None)
        self.assertEqual(rendered["title"], original)
        self.assertEqual(json.loads(rendered["json"])["answer"], original)
        self.assertNotIn("total_matches", json.loads(rendered["json"]))

    def test_review_cannot_manufacture_humans_or_weaken_quotas(self):
        row = {"block_id": "a"}
        for sizes in ({"train": 1, "validation": 1, "audit": 1}, {}):
            with self.assertRaises(ValueError):
                data.select_reviewed([row], [], sizes or {"train": 0, "validation": 0, "audit": 0})
        with self.assertRaises(ValueError):
            data.select_reviewed([row], [{"block_id": "a", "reviewer_kind": "human", "reason": "fake", "decision": "approve"}])
        with self.assertRaises(ValueError):
            data.select_reviewed([], [])

    def test_cross_original_overlap_is_rejected(self):
        row = {"entity_components": ["1", "2", "3"], "query_title": "query", "chosen_title": "chosen", "rejected_title": "reject", "human_pairwise_vote": False, "source_label": "relevant_vs_irrelevant"}
        self.assertEqual(data.check_splits({"train": [row]})["independent_groups"], {"train": 1})
        with self.assertRaises(ValueError):
            data.check_splits({"train": [row], "audit": [copy.deepcopy(row)]})

    def test_remote_neural_guard_without_imports(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            scorer.remote_only()
        self.assertNotIn("torch", __import__("sys").modules)
        self.assertNotIn("transformers", __import__("sys").modules)

    def test_validation_only_selection_and_earliest_tie(self):
        best, _ = scorer.select_epoch(None, 1, .5)
        best, changed = scorer.select_epoch(best, 2, .4)
        self.assertTrue(changed)
        best, changed = scorer.select_epoch(best, 3, .4)
        self.assertFalse(changed)
        self.assertEqual(best["epoch"], 2)
        for value in (math.nan, math.inf, None, True):
            with self.assertRaises(ValueError):
                scorer.select_epoch(None, 1, value)

    def test_binary_header_parser_failure_controls(self):
        for raw in (b"", struct.pack("<Q", 2 ** 40), struct.pack("<Q", 2) + b"[]", struct.pack("<Q", 2) + b"{}"):
            if raw.endswith(b"{}"):
                self.assertEqual(checkpoint.tensor_parts(raw), ({}, b""))
            else:
                with self.assertRaises(ValueError):
                    checkpoint.tensor_parts(raw)

    def test_cached_receipt_tampering_fails(self):
        with tempfile.TemporaryDirectory(dir=source.HERE) as tmp:
            path = Path(tmp) / "source.json"
            path.write_bytes(b"original")
            source.write(path.with_suffix(".json.receipt.json"), {"requested_url": source.VERSION, "bytes": 8, "sha256": source.sha(b"original")})
            self.assertEqual(source.fetch(source.VERSION, path), b"original")
            path.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                source.fetch(source.VERSION, path)

    def test_immutable_integer_baseline_controls(self):
        # Artificial control fixture, never exported as source training data.
        rows = [{"block_id": str(i), "query_title": "query alpha", "chosen_title": "alpha preferred", "rejected_title": "beta other"} for i in range(128)]
        fixed = baselines.fit(rows, "a" * 64)
        before = copy.deepcopy(fixed)
        self.assertEqual(baselines.replay(fixed, rows[:2], "a" * 64)["overlap"], 1.0)
        self.assertEqual(fixed, before)
        self.assertTrue(all(type(c) is int for c in fixed["answer_only_coefficients"]))
        for change in ({"fit_split": "audit"}, {"training_sha256": "b" * 64}, {"document_frequencies": [0] * len(fixed["vocabulary"])}):
            with self.assertRaises(ValueError):
                baselines.validate({**fixed, **change}, "a" * 64)

    def test_safetensors_licence_does_not_change_payload(self):
        with tempfile.TemporaryDirectory(dir=source.HERE) as tmp:
            path = Path(tmp) / "fixture.safetensors"
            header = json.dumps({"__metadata__": {"fixture": "container test only"}, "x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
            payload = struct.pack("<f", 1.0)
            path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
            checkpoint.preserve_artifact(path)
            parsed, after = checkpoint.tensor_parts(path.read_bytes())
            self.assertEqual(after, payload)
            self.assertEqual(parsed["__metadata__"]["x-licence"], "LicenseRef-zkasuran-SAND-1.0")
            before = path.read_bytes()
            checkpoint.preserve_artifact(path)
            self.assertEqual(path.read_bytes(), before)

    def test_unseen_terms_are_not_hidden_from_overlap_controls(self):
        query = "novelgene unrelatedcondition"
        self.assertGreater(overlap.signals(query, "novelgene pathway")["literal_shared_words"], overlap.signals(query, "othergene pathway")["literal_shared_words"])

    def test_one_training_format_per_original(self):
        rows = [{"block_id": str(i), "prompt": "query" + str(i), "formats": {f: {"chosen": f + str(i), "rejected": "wrong" + f + str(i)} for f in data.FORMATS}} for i in range(1, 20)]
        pairs = scorer.training_pairs(rows)
        self.assertEqual(len(pairs), len(rows))
        self.assertEqual(len({r["prompt"] for r in pairs}), len(rows))
        self.assertEqual(scorer.training_pairs(rows), pairs)

    def test_input_root_is_order_and_length_sensitive(self):
        self.assertNotEqual(wire.input_root([[1, 2], [3]]), wire.input_root([[3], [1, 2]]))
        self.assertNotEqual(wire.input_root([[1, 2]]), wire.input_root([[1, 2], []]))
        self.assertEqual(wire.input_root([]), "")

    def test_model_configuration_identity_and_gates_are_strict(self):
        from verify import validate_config
        config = json.loads((source.HERE / "model-contract.json").read_bytes())
        validate_config(config)
        for key, value in (("child", "OTHER_GENERAL"), ("maximum_workflow_minutes", 351), ("engine_sha256", "0" * 64), ("independent_originals", {"train": 1, "validation": 1, "audit": 1})):
            with self.assertRaises(ValueError):
                validate_config({**config, key: value})
        changed = copy.deepcopy(config)
        changed["hyperparameters"]["lora_rank"] = 16
        with self.assertRaises(ValueError):
            validate_config(changed)

    def test_remote_pipeline_entry_points_refuse_local_execution(self):
        import bootstrap
        import audit
        import pipeline
        import serving
        with patch.dict(os.environ, {}, clear=True):
            for function in (bootstrap.prepare_models, audit.audit, pipeline.train, serving.backend):
                with self.assertRaises(RuntimeError):
                    if function is audit.audit:
                        function(Path("unused"), Path("unused"))
                    else:
                        function(Path("unused"))
        self.assertNotIn("torch", __import__("sys").modules)

    def test_quality_rejects_each_floor_missing_format_and_nan(self):
        from report import quality
        good = {f: {"accuracy": .9, "per_label": {"chosen_longer": .8, "chosen_shorter": .9}, "untrained_accuracy": .5, "shortcut_accuracy": .55} for f in data.FORMATS}
        self.assertTrue(quality(good)["passed"])
        for field, value in (("accuracy", .79), ("accuracy", float("nan")), ("per_label", {"chosen_shorter": .69}), ("untrained_accuracy", .89), ("shortcut_accuracy", .85)):
            bad = copy.deepcopy(good)
            bad["json"][field] = value
            self.assertFalse(quality(bad)["passed"])
        self.assertFalse(quality({"title": good["title"]})["passed"])

    def test_ready_contract_cannot_turn_omitted_runtime_into_success(self):
        from report import ready, quality
        # Artificial report schema fixture only. Never saved as a real result.
        formats = {f: {"accuracy": .9, "per_label": {"shorter": .9}, "untrained_accuracy": .5, "shortcut_accuracy": .5} for f in data.FORMATS}
        value = {"family": "ACADEMIC_SEARCH", "run_id": 123, "commit": "a" * 40, "serving_commit": "a" * 40,
                 "package_sha256": "b" * 64, "package_keccak": "c" * 64,
                 "foundation_sha256": wire.FOUNDATION, "engine_sha256": wire.ENGINE, "tokenizer_sha256": wire.TOKENIZER,
                 "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True, "independent_groups": 64, "source_manifest_sha256": "d" * 64},
                 "quality": quality(formats), "runtime": {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
                 "negative_identity_controls": 3, "vectors": [{"tokens": [1, 2], "score_bits": 0x3f800000} for _ in range(8)]}}
        self.assertEqual(ready(value)["status"], "REGISTRATION_READY")
        for change in ({"runtime": {}}, {"source": {}}, {"package_keccak": "0x" + "c" * 64}, {"serving_commit": "e" * 40}, {"quality": {**value["quality"], "passed": False}}):
            with self.assertRaises(ValueError):
                ready({**value, **change})

    def test_public_serving_keeps_literal_hashes_and_original_token_root(self):
        from serving import public_request
        tokens = [1] * 512
        value = {"batch": [{"tokens": tokens}], "foundation_hash": wire.FOUNDATION, "adapter_hash": "a" * 64, "tokenizer_hash": wire.TOKENIZER}
        actual = public_request(value, "a" * 64)
        self.assertEqual(actual["batch"][0]["tokens"], tokens)
        self.assertEqual(actual["input_root"], wire.input_root([tokens]))
        for bad in ({**value, "adapter_hash": "0x" + "a" * 64}, {**value, "batch": [{"tokens": [1] * 513}]}, {**value, "batch": [{"tokens": [-1]}]}, {**value, "batch": [{"tokens": None}]}, {**value, "text": "untrusted answer"}):
            with self.assertRaises(ValueError):
                public_request(bad, "a" * 64)

    def test_verified_endpoint_reservation_is_bounded(self):
        from serving import validate_reservation, immutable_url
        validate_reservation(2, 48602, "https://tga1.zkasuran.dev/_academic_test/runnable")
        for slot, port, url in ((0, 48600, "https://tga1.zkasuran.dev/_x/runnable"), (2, 22, "https://tga1.zkasuran.dev/_x/runnable"), (2, 48602, "https://other.example/runnable")):
            with self.assertRaises(ValueError):
                validate_reservation(slot, port, url)
        with self.assertRaises(ValueError):
            immutable_url("https://raw.githubusercontent.com/zkasuran/telegraph-catalog-miners/main/file")

    def test_workflow_schema_refuses_paid_runner_and_fanout(self):
        from verify import validate_workflow
        path = source.HERE / "workflow.yml"
        if not path.exists():
            path = source.HERE.parents[2] / ".github/workflows/evaluator-owner-academic_search.yml"
        value = json.loads("\n".join(line for line in path.read_text().splitlines() if not line.startswith("#")))
        validate_workflow(value)
        for field, data_value in (("runs-on", "self-hosted"), ("timeout-minutes", 351), ("strategy", {"matrix": {"copies": list(range(7))}})):
            bad = copy.deepcopy(value)
            bad["jobs"]["academic"][field] = data_value
            with self.assertRaises(ValueError):
                validate_workflow(bad)

    def test_public_source_download_refuses_untrusted_host_without_network(self):
        from bootstrap import fetch
        with tempfile.TemporaryDirectory(dir=source.HERE) as temp:
            with self.assertRaises(ValueError):
                fetch("http://127.0.0.1/private", Path(temp) / "data", "a" * 64)

    def test_verified_publisher_metadata_redirect_is_allowed_but_not_http_or_other_host(self):
        import urllib.request
        from bootstrap import SafeRedirect
        handler = SafeRedirect()
        request = urllib.request.Request("https://huggingface.co/answerdotai/ModernBERT-base/resolve/pin/config.json")
        redirected = handler.redirect_request(request, None, 302, "Found", {}, "https://huggingface.co/api/resolve-cache/models/answerdotai/ModernBERT-base/pin/config.json")
        self.assertEqual(redirected.host, "huggingface.co")
        for url in ("http://huggingface.co/config.json", "https://127.0.0.1/config.json", "https://untrusted.example/config.json"):
            with self.assertRaises(ValueError):
                handler.redirect_request(request, None, 302, "Found", {}, url)

    def test_dependency_auditor_does_not_skip_unknown_variants(self):
        from dependency_audit import requirements
        parsed, mapped = requirements("torch==2.15.0+cpu\nnumpy==2.4.0\n")
        self.assertIn("torch==2.15.0", parsed)
        self.assertIn("torch", mapped)
        with self.assertRaises(ValueError):
            requirements("other==1.0+private\n")


if __name__ == "__main__":
    unittest.main()
