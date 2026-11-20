# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Fake/static controls only. No local neural model or fitted training data."""
import copy
import json
import os
import sys
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import patch

from baselines import scores
from contracts import CHILD, EXPERIMENT, FAMILY, FORMATS, FOUNDATION, bits, canonical, check_class, check_reply, eligibility, identity, input_root, validate_experiment, validate_report
from transport import RuntimeBridge, live_controls


class OwnerTests(unittest.TestCase):
    def test_fitted_warm_start_never_becomes_untrained_baseline(self):
        warm = EXPERIMENT["warm_start"]
        self.assertNotEqual(warm["checkpoint_sha256"], warm["untrained_sha256"])
        self.assertIn("Fresh AdamW", warm["optimizer"])
        changed = copy.deepcopy(EXPERIMENT)
        changed["warm_start"]["untrained_sha256"] = "not-a-hash"
        with self.assertRaises(ValueError):
            validate_experiment(changed)
    def test_preserved_rng_isolation_with_fake_success_and_failure(self):
        from remote import support
        isolated = support()["core"].isolated_rng
        class State(int):
            def clone(self):
                return State(self)
        class FakeTorch:
            state = 7
            def get_rng_state(self):
                return State(self.state)
            def equal(self, a, b):
                return a == b
            @property
            def random(self):
                return self
            @contextmanager
            def fork_rng(self, devices):
                self.devices = devices
                before = self.state
                try:
                    yield
                finally:
                    if self.restore:
                        self.state = before
        class FakeNumpy:
            state = (1, 2)
            @property
            def random(self):
                return self
            def get_state(self):
                return self.state
            def set_state(self, value):
                self.state = value
        torch, np = FakeTorch(), FakeNumpy()
        torch.restore = True
        with isolated(torch, np):
            torch.state += 1
            np.state = (2, 3)
        self.assertEqual(torch.state, 7)
        self.assertEqual(torch.devices, [])
        self.assertEqual(np.state, (1, 2))
        torch.restore = False
        with self.assertRaisesRegex(RuntimeError, "Torch RNG"), isolated(torch, np):
            torch.state += 1

    def test_refinement_order_plan_and_exact_boundary_failure_controls(self):
        from remote import support
        matched = support()["matched"]
        rows = [{"group_id": str(i)} for i in range(4)]
        plan = {"source_groups": [str(i) for i in range(4)], "epochs": [[0, 1, 2, 3] for _ in range(3)]}
        matched.validate_orders(plan, rows)
        bad = {**plan, "epochs": plan["epochs"][:2]}
        with self.assertRaises(ValueError):
            matched.validate_orders(bad, rows)
        fields = support()["core"].CONTRACT["matched_trial"]["matched_fields"]
        expected = {field: "fake-fixed" for field in fields}
        self.assertTrue(matched.require_match(expected, dict(expected))["exact_match"])
        for field in fields:
            bad = {**expected, field: "fake-changed"}
            with self.assertRaises(ValueError):
                matched.require_match(expected, bad)

    def metrics(self, accuracy=0.9):
        return {name: {"accuracy": accuracy, "by_label": {label: {"accuracy": accuracy} for label in ("Flagged", "Safe")}} for name in FORMATS}

    def report(self):
        return {"schema_version": 1, "status": "REGISTRATION_READY", "family": FAMILY, "run_id": 123,
                "commit": "b" * 40, "serving_commit": "b" * 40, "package_bytes": 32,
                "package_sha256": "a" * 64, "package_keccak": "c" * 64,
                "foundation_sha256": FOUNDATION["foundation_lora_fp32.onnx"], "engine_sha256": FOUNDATION["foundation_engine.wasm"], "tokenizer_sha256": FOUNDATION["tokenizer.json"],
                "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True, "independent_groups": 256, "source_manifest_sha256": "d" * 64},
                "quality": {"passed": True, "audit_pairs": 256, "all_supported_formats_tested": True,
                            "formats": {f: {"accuracy": 0.9, "per_label": {"Flagged": 0.9, "Safe": 0.9}, "untrained_accuracy": 0.5, "shortcut_accuracy": 0.5} for f in FORMATS}},
                "runtime": {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True, "negative_identity_controls": 3,
                            "vectors": [{"tokens": [50281, 42, 50282], "score_bits": 0x3f800000} for _ in range(8)]},
                "test_scope": "Synthetic report contract control only, not a measured result."}

    def test_frozen_experiment_parses_and_refuses_recipe_changes(self):
        validate_experiment(copy.deepcopy(EXPERIMENT))
        for field in ("epochs", "audit_originals", "job_minutes", "historical_comparison_valid"):
            changed = copy.deepcopy(EXPERIMENT)
            changed[field] = 0
            with self.assertRaises(ValueError):
                validate_experiment(changed)
        changed = copy.deepcopy(EXPERIMENT)
        changed["hyperparameters"]["lr"] = 0.01
        with self.assertRaises(ValueError):
            validate_experiment(changed)

    def test_quality_pass_and_each_failure_control(self):
        good, baseline, shortcuts = self.metrics(), self.metrics(0.5), {f: 0.5 for f in FORMATS}
        self.assertEqual(eligibility(good, baseline, shortcuts), [])
        changed = copy.deepcopy(good)
        changed["json"]["by_label"]["Flagged"]["accuracy"] = 0.69
        self.assertTrue(eligibility(changed, baseline, shortcuts))
        changed = copy.deepcopy(good)
        del changed["bare_label"]
        self.assertTrue(eligibility(changed, baseline, shortcuts))
        self.assertTrue(eligibility(good, self.metrics(0.86), shortcuts))
        self.assertTrue(eligibility(good, baseline, {f: 0.85 for f in FORMATS}))
        self.assertTrue(eligibility(self.metrics(float("nan")), baseline, shortcuts))
        changed = self.metrics()
        changed["json"]["accuracy"] = 102 / 128
        self.assertTrue(eligibility(changed, baseline, shortcuts))  # Real H9 boundary must not be rounded into a pass.

    def test_actual_finite_bits_contract(self):
        self.assertEqual(bits(0x3f800000), 1.0)
        for value in (True, -1, 1 << 32, 0x7f800000, 0x7fc00000):
            with self.assertRaises(ValueError):
                bits(value)

    def test_literal_wire_not_normalized_standin(self):
        package = "c" * 64
        request = {"batch": [{"tokens": [50281, 42, 50282]}], **identity(package)}
        request["input_root"] = input_root(request["batch"])
        response = {"results": [{"score": 0x3f800000}], **identity(package), "input_root": request["input_root"]}
        self.assertEqual(check_reply(response, request, package), [0x3f800000])
        for value in ("0x" + package, package.upper(), " " + package):
            with self.assertRaises(ValueError):
                check_reply({**response, "adapter_hash": value}, request, package)
        with self.assertRaises(ValueError):
            check_reply({**response, "results": []}, request, package)

    def test_primary_http_smoke_fake_backend_and_real_refusal_routes(self):
        package = "c" * 64
        class FakeBackend:
            def score(self, request):
                return {"results": [{"score": 0x3f800000} for _ in request["batch"]],
                        **identity(package), "adapter_hash": "0x" + package,
                        "input_root": request["input_root"]}
        vectors, controls = live_controls(RuntimeBridge(FakeBackend(), package), package)
        self.assertEqual(len(vectors), 8)
        self.assertEqual(controls, 3)
        self.assertTrue(all(v["score_bits"] == 0x3f800000 for v in vectors))

    def test_frozen_baselines_repeat_without_fitting_or_numpy(self):
        coefficient = {"idf": {"text": 1.0}, "answer_only": {"weights": {"safe": 1.0, "flagged": 1.0}},
                       "directions": {name: 1 for name in ("length", "tfidf_overlap", "alarm_keywords", "answer_only")}}
        rows = [{"prompt": "text", "chosen": "Safe", "rejected": "Flagged"}, {"prompt": "text", "chosen": "Flagged", "rejected": "Safe"}]
        frozen = canonical(coefficient)
        self.assertEqual(scores(rows, coefficient), scores(rows, json.loads(frozen)))
        self.assertEqual(canonical(coefficient), frozen)
        self.assertNotIn("numpy", sys.modules)

    def test_local_model_calls_refuse_before_neural_imports(self):
        import baselines
        import remote
        import serve
        with patch.dict(os.environ, {}, clear=True):
            for call in (lambda: baselines.fit([]), lambda: remote.export_package(None, None, None), lambda: remote.engine_audit(None, None, None, None), lambda: serve.hold(None)):
                with self.assertRaises(RuntimeError):
                    call()
        for name in ("torch", "transformers", "peft", "tokenizers", "numpy"):
            self.assertNotIn(name, sys.modules)

    def test_report_gate_never_accepts_missing_actual_runtime_or_labels(self):
        report = self.report()
        self.assertEqual(len(validate_report(report)), 8)
        for key in ("full_parser_passed", "reference_engine_passed", "repeat_bits_identical"):
            changed = copy.deepcopy(report)
            changed["runtime"][key] = False
            with self.assertRaises(ValueError):
                validate_report(changed)
        changed = copy.deepcopy(report)
        changed["quality"]["formats"]["json"]["per_label"]["Flagged"] = 0.69
        with self.assertRaises(ValueError):
            validate_report(changed)
        changed = copy.deepcopy(report)
        changed["source"]["audit_used_for_selection"] = True
        with self.assertRaises(ValueError):
            validate_report(changed)
        changed = copy.deepcopy(report)
        changed["runtime"]["vectors"][0]["score_bits"] = 0x7fc00000
        with self.assertRaises(ValueError):
            validate_report(changed)

    def test_authoritative_class_and_swapped_family_refusal(self):
        child = {"intent_id": CHILD, "intent_name": CHILD, "class": "judgment", "evaluation": {"kind": "judgment"},
                 "request": {"served_by": FAMILY}, "response": {"served_by": FAMILY}, "onchain": {"served_by": FAMILY, "ready": True}}
        source = {"schema": 1, "fetched_at": datetime.now(timezone.utc).isoformat(),
                  "family": {"url": "https://devnode.telegraphprotocol.com/api/wasm?resolve=" + FAMILY, "raw": json.dumps({"resolve": {"intent": FAMILY, "class": "judgment"}})},
                  "child": {"url": "https://devnode.telegraphprotocol.com/engine/v1/intents/" + CHILD, "raw": json.dumps(child)}}
        check_class(source)
        child["request"]["served_by"] = "SENTIMENT_ANALYSIS"
        source["child"]["raw"] = json.dumps(child)
        with self.assertRaises(ValueError):
            check_class(source)


if __name__ == "__main__":
    unittest.main()
