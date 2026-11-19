# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
import copy
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import data
import scorer
import runtime
import remote

HERE = Path(__file__).resolve().parent


def answer(code=0, name="_dmarc.unit.test.", kind=16, **extra):
    return json.dumps({"Status": code, "Question": [{"name": name, "type": kind}], **extra})


class ScorerTests(unittest.TestCase):
    def setUp(self):
        self.prompt = scorer.request_prompt("_DMARC.Unit.test", "TXT")

    def test_literal_rendering(self):
        literal = ("EMAIL_SECURITY_GENERAL request. domain (Email domain): _DMARC.Unit.test. "
                   "record_type (DNS record type to look up, e.g. MX or TXT): TXT.")
        self.assertEqual(self.prompt, literal)
        self.assertEqual(scorer.render(literal, "raw"), "### Task:\n" + literal + "\n\n### Answer:\nraw")

    def test_noerror_nodata(self):
        result = scorer.inspect(self.prompt, answer(Answer=[]))
        self.assertTrue(result["lookup_ok"])
        self.assertEqual(result["score"], 1)
        self.assertFalse(result["externally_authenticated"])

    def test_null_mx_stays_lookup_success(self):
        result = scorer.inspect(scorer.request_prompt("unit.test", "MX"),
                                answer(name="unit.test.", kind=15, Answer=[{"type": 15, "data": "0 ."}]))
        self.assertTrue(result["lookup_ok"])

    def test_every_defined_rcode(self):
        for code in range(16):
            with self.subTest(code=code):
                result = scorer.inspect(self.prompt, answer(code))
                self.assertEqual(result["lookup_ok"], code == 0)

    def test_rcode_strings(self):
        for label, code in scorer.RCODES.items():
            self.assertEqual(scorer.inspect(self.prompt, answer(label))["lookup_ok"], code == 0)

    def test_contradictory_claim(self):
        self.assertEqual(scorer.inspect(self.prompt, answer(0, lookup_ok=False))["score"], -1)
        self.assertEqual(scorer.inspect(self.prompt, answer(3, lookup_ok=True))["score"], -1)

    def test_question_name_identity(self):
        self.assertEqual(scorer.inspect(self.prompt, answer(name="_dmarc.other.test."))["score"], -1)
        self.assertEqual(scorer.inspect(self.prompt, answer(name="unit.test."))["score"], -1)
        self.assertEqual(scorer.inspect(self.prompt, answer(name="_dmarc.unit.test.attacker.test."))["score"], -1)

    def test_type_identity(self):
        self.assertEqual(scorer.inspect(self.prompt, answer(kind=15))["score"], -1)
        self.assertEqual(scorer.inspect(self.prompt, answer(kind="TXT"))["score"], 1)

    def test_inconclusive_not_false(self):
        for body in ({"lookup_ok": False}, {"error": "timeout"}, {"Status": 0},
                     {"Status": 0, "Question": []}, {"Status": 0, "Question": [{}, {}]}):
            self.assertIsNone(scorer.inspect(self.prompt, json.dumps(body))["score"])

    def test_http_and_dnssec_not_status(self):
        self.assertFalse(scorer.inspect(self.prompt, answer(3, http_status=200, AD=True))["lookup_ok"])

    def test_invalid_status_types(self):
        for code in (True, -1, 4096, 0.0, "0", "timeout", None):
            with self.subTest(code=code), self.assertRaises(ValueError):
                scorer.inspect(self.prompt, answer(code))

    def test_duplicate_keys_and_json_bounds(self):
        bad = ('{"Status":3,"Status":0}', '[]', '{"a":NaN}', '{"a":1e999}', '{' * 20,
               '{"a":' + '[' * 13 + '0' + ']' * 13 + '}', 'x' * 16385)
        for raw in bad:
            with self.subTest(raw=raw[:50]), self.assertRaises(ValueError):
                scorer.parse_json(raw)

    def test_domain_boundaries(self):
        self.assertEqual(scorer.domain("_DMARC.Unit.test."), "_dmarc.unit.test")
        for name in ("https://unit.test", "unit.test ", "a..test", "unit.test..", "é.test", "a" * 64 + ".test"):
            with self.assertRaises(ValueError):
                scorer.domain(name)

    def test_hostile_random_bytes(self):
        rng = random.Random(722)
        for _ in range(2000):
            raw = rng.randbytes(rng.randrange(0, 128))
            try:
                scorer.parse_json(raw.decode("utf-8"))
            except (ValueError, UnicodeError):
                pass

    def test_primary_cli_smoke_and_failure(self):
        args = [sys.executable, str(HERE / "scorer.py"), "--domain", "_dmarc.unit.test", "--record-type", "TXT"]
        good = subprocess.run(args, input=answer(), text=True, capture_output=True, timeout=10, check=False)
        self.assertEqual(good.returncode, 0)
        self.assertEqual(json.loads(good.stdout)["score"], 1)
        bad = subprocess.run(args, input=answer(name="other.test."), text=True, capture_output=True, timeout=10, check=False)
        self.assertEqual(bad.returncode, 1)
        malformed = subprocess.run(args, input='{"Status":true}', text=True, capture_output=True, timeout=10, check=False)
        self.assertEqual(malformed.returncode, 2)

    def test_failure_controls_kill_mutants(self):
        # Each mutation changes the actual dependency the scorer consumes.
        with patch.object(scorer, "status", return_value=0):
            self.assertNotEqual(scorer.inspect(self.prompt, answer(3))["lookup_ok"], False)
        with patch.object(scorer, "domain", return_value="constant.test"):
            self.assertNotEqual(scorer.inspect(self.prompt, answer(name="other.test."))["score"], -1)
        with patch.object(scorer, "record_type", return_value=16):
            self.assertNotEqual(scorer.inspect(self.prompt, answer(kind=15))["score"], -1)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.splits = {name: data.make_groups(name) for name in data.SIZES}

    def test_full_group_inventory(self):
        result = data.check_groups(self.splits)
        self.assertEqual(result["groups"], 224)
        self.assertEqual(result["normalized_split_overlaps"], 0)
        self.assertEqual(result["original_observed_dns_records"], 0)

    def test_counterbalance_and_frozen_train_only(self):
        frozen = data.freeze_controls(self.splits["train"])
        self.assertTrue(all(value == 0 for value in frozen["coefficients"].values()))
        controls = data.evaluate_controls(self.splits["validation"], frozen)
        self.assertEqual(controls["answer_only"], .5)
        self.assertEqual(controls["length"], .5)
        self.assertEqual(controls["exact_question_parser"], 1)
        changed = copy.deepcopy(self.splits)
        changed["audit"][0]["chosen"] = "audit cannot influence coefficients"
        self.assertEqual(frozen, data.freeze_controls(changed["train"]))

    def test_overlap_fails(self):
        changed = copy.deepcopy(self.splits)
        changed["validation"][:2] = changed["train"][:2]
        with self.assertRaisesRegex(ValueError, "overlap"):
            data.check_groups(changed)

    def test_missing_counterbalance_fails(self):
        changed = copy.deepcopy(self.splits)
        changed["train"][1]["group_id"] = "new-unpaired-group"
        with self.assertRaisesRegex(ValueError, "counterbalance"):
            data.check_groups(changed)

    def test_hash_binding_and_reproduction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = data.prepare(root)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            self.assertEqual(report, data.prepare(root))
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
            data.read_splits(root)
            with (root / "train.jsonl").open("ab") as output:
                output.write(b" ")
            with self.assertRaisesRegex(ValueError, "hash"):
                data.read_splits(root)
            with self.assertRaisesRegex(ValueError, "replace"):
                data.prepare(root)

    def test_remote_guard_before_neural_imports(self):
        spec = importlib.util.spec_from_file_location("email_remote_test", HERE / "remote.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(RuntimeError, "public CPU"):
            module.remote_only()
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)


class RuntimeBoundaryTests(unittest.TestCase):
    def test_literal_response_contract_and_identity_failures(self):
        manifest = {"keccak256": "0x" + "a" * 64}
        value = {"foundation_hash": runtime.HASHES["foundation_lora_fp32.onnx"],
                 "tokenizer_hash": runtime.HASHES["tokenizer.json"], "adapter_hash": "a" * 64,
                 "results": [{"score": 0x3f800000}]}
        self.assertEqual(runtime.check_response(value, 1, manifest), [0x3f800000])
        for key in ("foundation_hash", "tokenizer_hash", "adapter_hash"):
            bad = {**value, key: "0" * 64}
            with self.assertRaisesRegex(ValueError, "identity"):
                runtime.check_response(bad, 1, manifest)

    def test_runtime_result_failure_controls(self):
        for value in (True, -1, 1 << 32, 0x7fc00000, 0x7f800000):
            with self.assertRaises(ValueError):
                runtime.decode_bits(value)
        for raw in ([], {}, {"results": []}):
            with self.assertRaises(ValueError):
                runtime.check_response(raw, 1, {"keccak256": "a" * 64})

    def test_recipe_refuses_unbounded_or_foreign_scope(self):
        recipe = json.loads((HERE / "recipe.json").read_bytes())
        remote.validate_recipe(recipe)
        for key, replacement in (("family", "OTHER"), ("rank", 16), ("epochs", 3),
                                 ("runner", "self-hosted"), ("training_timeout_seconds", 7201),
                                 ("production_transport_verified", True)):
            with self.assertRaises(ValueError):
                remote.validate_recipe({**recipe, key: replacement})

    def test_preservation_excludes_downloaded_upstream(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("adapter", "checkpoint", "base", "foundation", "source-kit"):
                (root / name).mkdir()
                (root / name / "test.safetensors").write_bytes(b"container-test-only")
            called = []
            with patch("licensing.preserve_artifact", side_effect=lambda path: called.append(path.relative_to(root).parts[0])):
                remote.preserve(root)
            self.assertEqual(sorted(called), ["adapter", "checkpoint"])


if __name__ == "__main__":
    unittest.main()
