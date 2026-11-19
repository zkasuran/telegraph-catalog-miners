# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only boundary tests. Authored fixtures are not training examples."""
import copy
import http.client
import json
import os
import random
import threading
import unittest
from unittest.mock import patch

from baselines import fit, measure, validate_frozen
from balanced import allocate
from contracts import FAMILY, FOUNDATION, HERE, check_config, check_wire, decode_bits, quality, remote_only, render, strict_json
from data import normalized, pair, source_group
from live import validate
from reference_render import render as actual_reference_render
from executor import Server, validate_request
from serve_remote import immutable_url
from validation_policy import require_validation, warmstart_policy
from parent_checkpoint import restore
from source_cues import measure_source_cues


def fixture(label="Entailment", hypothesis_id="control"):
    return pair({"document_id": "authored-parser-control", "hypothesis_id": hypothesis_id,
                 "hypothesis": "Receiving Party may retain records as required by law.",
                 "evidence": "Recipient may retain a record when required by applicable law.",
                 "evidence_sha256": "control-not-a-source-hash", "label": label})


class ContractTests(unittest.TestCase):
    def test_parent_restore_never_loads_locally(self):
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false", "OWNER_SOURCE_ONLY": "1"}, clear=True):
            with self.assertRaises(RuntimeError):
                restore(HERE)

    def test_source_aware_baseline_is_separate_and_frozen(self):
        row = fixture("Contradiction")
        row["evidence"] = "No recipient shall act without written consent except as required by law."
        frozen = {"fit_split": "train", "integer_coefficients": {"negation": {"threshold": 2, "direction": -1},
                   "exception": {"threshold": 1, "direction": 1}}}
        measured = measure_source_cues([row], frozen)
        self.assertEqual(measured["negation"], 1.0)
        self.assertEqual(measured["exception"], 0.0)
        frozen["fit_split"] = "audit"
        with self.assertRaises(ValueError):
            measure_source_cues([row], frozen)
    def test_validation_failure_keeps_audit_locked(self):
        # Actual failed run produced only this incomplete single-format summary.
        observed = {"pairwise_accuracy": 0.5, "n_eval": 16, "verdict": "FAIL", "val_loss": 0.692055}
        with self.assertRaises(ValueError):
            require_validation(observed, 16)
        cases = {name: {"accuracy": .5, "per_label": {"Entailment": .5, "Contradiction": .5},
                        "untrained_accuracy": .5, "shortcut_accuracy": .5, "original_pairs": 16,
                        "fit_split": "validation"} for name in ("json", "prose")}
        with self.assertRaises(ValueError):
            require_validation(cases, 16)
        for metrics in cases.values():
            metrics["accuracy"] = .875
            metrics["per_label"] = {"Entailment": .875, "Contradiction": .875}
        self.assertEqual(require_validation(cases, 16)["status"], "VALIDATION_READY")
        cases["prose"]["per_label"]["Contradiction"] = .625
        with self.assertRaises(ValueError):
            require_validation(cases, 16)

    def test_warmstart_policy_retains_gates(self):
        policy = warmstart_policy()
        self.assertEqual(policy["epochs"], 32)
        self.assertEqual(policy["train_formats"], ["json", "prose"])
        self.assertIn("retained licensed", policy["initialization"])
        self.assertIn("80%", policy["quality_gate"])
    def test_exact_render(self):
        self.assertEqual(render(" Café\u2028\n", " Unclear "), "### Task:\n Café\u2028\n\n\n### Answer:\n Unclear ")
        self.assertNotEqual(render(" q ", "a"), render("q", "a"))
        for prompt, answer in ((" Café\u2028\n", " Unclear "), ('{"contract":"A"}', '{"verdict":"Unclear"}')):
            self.assertEqual(render(prompt, answer), actual_reference_render(prompt, answer))

    def test_strict_duplicate_json(self):
        with self.assertRaises(ValueError):
            strict_json('{"verdict":"Unclear","verdict":"Compliant"}')

    def test_nonfinite_json_refused(self):
        for raw in ("NaN", "Infinity", "-Infinity"):
            with self.assertRaises(ValueError):
                strict_json(raw)

    def test_finite_score_bits(self):
        self.assertEqual(decode_bits(0x3f800000), 1.0)
        self.assertEqual(decode_bits(0xbf800000), -1.0)
        for raw in (True, -1, 1 << 32, 0x7f800000, 0x7fc00001, "1065353216"):
            with self.assertRaises(ValueError):
                decode_bits(raw)

    def test_wire_identity_literal_bare_hash(self):
        expected = "a" * 64
        response = {"adapter_hash": expected, "foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
                    "tokenizer_hash": FOUNDATION["tokenizer.json"], "results": [{"score": 0x3f800000}]}
        self.assertEqual(check_wire(response, 1, expected), [0x3f800000])
        for field in ("adapter_hash", "foundation_hash", "tokenizer_hash"):
            wrong = copy.deepcopy(response)
            wrong[field] = "0x" + wrong[field]
            with self.assertRaises(ValueError):
                check_wire(wrong, 1, expected)

    def test_wrong_wire_count(self):
        response = {"adapter_hash": "a" * 64, "foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
                    "tokenizer_hash": FOUNDATION["tokenizer.json"], "results": []}
        with self.assertRaises(ValueError):
            check_wire(response, 1, "a" * 64)

    def test_random_score_bits_no_unbounded_exception(self):
        rng = random.Random(42)
        for _ in range(500):
            bits = rng.randrange(1 << 32)
            if bits & 0x7f800000 == 0x7f800000:
                with self.assertRaises(ValueError):
                    decode_bits(bits)
            else:
                self.assertIsInstance(decode_bits(bits), float)

    def test_local_neural_guard(self):
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false", "OWNER_SOURCE_ONLY": "1"}, clear=True):
            with self.assertRaises(RuntimeError):
                remote_only()
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "other/repo", "GITHUB_EVENT_NAME": "workflow_dispatch"}, clear=True):
            with self.assertRaises(RuntimeError):
                remote_only()

    def test_config_parses_and_validates(self):
        config = strict_json((HERE / "config.json").read_bytes())
        check_config(config)
        for field, value in (("family", "OTHER_FAMILY"), ("max_tokens", 513), ("epochs", 999)):
            wrong = copy.deepcopy(config)
            wrong[field] = value
            with self.assertRaises(ValueError):
                check_config(wrong)

    def test_normalized_source_identity_not_wire(self):
        self.assertEqual(normalized("Ａ\n B"), normalized("a b"))

    def test_entity_source_groups(self):
        self.assertEqual(source_group({"url": "https://www.sec.gov/Archives/edgar/data/123/1/x.htm"}), "sec-cik:123")
        self.assertNotEqual(source_group({"url": "https://www.sec.gov/Archives/edgar/data/123/1/x.htm"}),
                            source_group({"url": "https://www.sec.gov/Archives/edgar/data/124/1/x.htm"}))
        self.assertEqual(source_group({"url": "https://www.example.org/a.pdf"}), source_group({"url": "https://example.org/b.pdf"}))

    def test_pair_keeps_source_evidence_and_unclear_verdict(self):
        row = fixture()
        chosen, rejected = json.loads(row["chosen"]), json.loads(row["rejected"])
        self.assertEqual(chosen["clauses"], rejected["clauses"])
        self.assertEqual(chosen["verdict"], "Unclear")
        self.assertEqual(rejected["verdict"], "Unclear")
        self.assertNotEqual(chosen["summary"], rejected["summary"])
        self.assertEqual(chosen["obligations"][0]["status"], "supports")
        self.assertEqual(rejected["obligations"][0]["status"], "refutes")
        self.assertIn("not fresh human", row["authorship"])

    def test_unclaimed_bare_label_is_not_a_fabricated_audit(self):
        with self.assertRaises(ValueError):
            pair(fixture(), "bare_label")

    def test_frozen_baseline_has_no_float_refitting(self):
        train = [fixture("Entailment"), fixture("Contradiction")]
        frozen = fit(train)
        self.assertEqual(frozen["bow_integer_coefficients"], {})
        validate_frozen(frozen, train)
        self.assertEqual(measure(train, frozen)["answer_only_bow"], .5)
        wrong = copy.deepcopy(frozen)
        wrong["fit_rows"] += 1
        with self.assertRaises(ValueError):
            validate_frozen(wrong, train)

    def test_hypothesis_prior_detects_missed_shortcut(self):
        train = [fixture("Entailment", "first") for _ in range(4)] + [fixture("Contradiction", "second") for _ in range(4)]
        frozen = fit(train)
        measured = measure(train, frozen)
        self.assertEqual(measured["answer_only_bow"], .5)
        self.assertEqual(measured["hypothesis_prior"], 1.0)

    def test_quality_improvement_unchanged(self):
        def report():
            return {name: {"accuracy": .85, "per_label": {"Entailment": .8, "Contradiction": .9},
                           "untrained_accuracy": .5, "shortcut_accuracy": .6} for name in ("json", "prose")}
        self.assertTrue(quality(report())["passed"])
        for key, value in (("accuracy", .79), ("untrained_accuracy", .81), ("shortcut_accuracy", .76)):
            data = report()
            data["json"][key] = value
            self.assertFalse(quality(data)["passed"])
        data = report()
        data["prose"]["per_label"]["Contradiction"] = .69
        self.assertFalse(quality(data)["passed"])
        with self.assertRaises(ValueError):
            quality({"json": report()["json"]})

    def test_primary_path_smoke(self):
        row = fixture("Contradiction")
        frozen = fit([row])
        self.assertEqual(measure([row], frozen)["hypothesis_prior"], 1.0)
        self.assertIn("refutes", json.loads(row["chosen"])["summary"])
        self.assertIn("supports", json.loads(row["rejected"])["summary"])

    def test_live_schema_failure_controls(self):
        snapshot = {"family_raw": json.dumps({"resolve": {"intent": FAMILY, "class": "judgment"}}),
                    "child_raw": json.dumps({"class": "judgment", "intent_id": FAMILY + "_GENERAL",
                        "request": {"params": [{"name": "contract"}]}, "onchain": {"ready": True},
                        "response": {"fields": [{"name": "verdict", "allowed": ["Compliant", "Partially compliant", "Non-compliant", "Unclear"]},
                        {"name": "clauses", "off_chain": True}, {"name": "obligations", "off_chain": True}]}})}
        validate(snapshot)
        wrong = copy.deepcopy(snapshot)
        wrong["family_raw"] = json.dumps({"resolve": {"intent": FAMILY, "class": "deterministic"}})
        with self.assertRaises(ValueError):
            validate(wrong)

    def test_balanced_pair_feasibility(self):
        def row(identifier, label, group):
            return {"document_id": identifier, "hypothesis_id": "control", "label": label,
                    "source_group": group, "normalized_evidence_sha256": identifier}
        pool = [row("a", "Contradiction", "one"), row("b", "Entailment", "one"),
                row("c", "Contradiction", "two"), row("d", "Entailment", "three")]
        rows = allocate(pool, ["control"], 4, 2)
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(row["label"] == "Entailment" for row in rows), 2)
        with self.assertRaises(ValueError):
            allocate(pool, ["control"], 6, 2)

    def test_immutable_serving_url(self):
        url = "https://raw.githubusercontent.com/zkasuran/telegraph-catalog-miners/"+"a"*40+"/verification/owners/contract_obligation_audit/results/package.safetensors"
        self.assertEqual(immutable_url(url), url)
        for wrong in (url+"?key=untrusted", url.replace("https", "http"), url.replace("results/package", "../package")):
            with self.assertRaises(ValueError):
                immutable_url(wrong)

    def test_real_http_frontend_literal_and_failure_controls(self):
        # Authored boundary fixture only. This is not a neural result or vector.
        class FixtureBackend:
            package_keccak = "a"*64
            def score(self, value):
                validate_request(value, self.package_keccak)
                return {"results": [{"score": 0x3f800000} for _ in value["batch"]],
                        **{key: value[key] for key in ("foundation_hash", "adapter_hash", "tokenizer_hash")}}
        backend = FixtureBackend()
        server = Server(("127.0.0.1", 0), backend)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = {"batch": [{"tokens": [50281, 22, 50282]}], "foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
                "adapter_hash": backend.package_keccak, "tokenizer_hash": FOUNDATION["tokenizer.json"]}
        try:
            cases = [(base, 200)]
            for key in ("foundation_hash", "adapter_hash", "tokenizer_hash"):
                wrong = copy.deepcopy(base)
                wrong[key] = "0x"+wrong[key]
                cases.append((wrong, 400))
            for tokens in ([], [-1], [True], [50368], [22]*513):
                wrong = copy.deepcopy(base)
                wrong["batch"] = [{"tokens": tokens}]
                cases.append((wrong, 400))
            for value, expected in cases:
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                try:
                    connection.request("POST", "/runnable", json.dumps(value), {"Content-Type": "application/json"})
                    response = connection.getresponse()
                    raw = response.read()
                    self.assertEqual(response.status, expected)
                    if expected == 200:
                        self.assertEqual(check_wire(json.loads(raw), 1, backend.package_keccak), [0x3f800000])
                    else:
                        self.assertEqual(json.loads(raw), {"error": "request refused"})
                finally:
                    connection.close()
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
