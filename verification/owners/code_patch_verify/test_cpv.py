# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
import ast
import copy
import json
import random
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from ci_reference import EvidenceError, collect_jobs, expected, render, run_identity, score, strict_json
from data import HERE, SPLITS, check_splits, control_metrics, fixture, freeze_controls
from remote import caller_contract, remote_only, stamp_checkpoint
from static_check import validate_recipe


class CIReferenceTests(unittest.TestCase):
    def setUp(self):
        self.row = fixture("train", 0)
        self.source = json.loads(self.row["prompt"])
        self.request = {"patch": self.source["patch"]}
        self.evidence = self.source["ci_observation"]

    def test_primary_pass_and_failure_control(self):
        self.assertEqual(score(self.request, self.row["chosen"], self.evidence)["score"], 1)
        self.assertEqual(score(self.request, self.row["rejected"], self.evidence)["score"], 0)

    def test_supported_passed_mapping_and_unresolved_states(self):
        for name in ("success", "failure", "cancelled", "skipped", "neutral", "timed_out", "pending"):
            evidence = copy.deepcopy(self.evidence)
            evidence["run"].update(status="queued" if name == "pending" else "completed",
                                    conclusion=None if name == "pending" else name)
            if name in ("success", "failure"):
                truth = expected(self.request, evidence)
                self.assertEqual(truth["passed"], name == "success")
                self.assertEqual(truth["conclusion"], name)
            else:
                result = score(self.request, self.row["chosen"], evidence)
                self.assertFalse(result["available"])
                self.assertIsNone(result["score"])

    def test_transport_200_is_not_ci_success(self):
        row = fixture("train", 1)
        source = json.loads(row["prompt"])
        self.assertFalse(expected({"patch": source["patch"]}, source["ci_observation"])["passed"])

    def test_attempt_pinning_no_latest_fallback(self):
        evidence = copy.deepcopy(self.evidence)
        evidence["run"]["run_attempt"] = 2
        result = score(self.request, self.row["chosen"], evidence)
        self.assertFalse(result["available"])
        self.assertIsNone(result["score"])

    def test_source_identity_and_commit_mismatch(self):
        for key, value in (("id", 456), ("head_sha", "bad"), ("repository", {"full_name": "other/repo"}),
                           ("html_url", "https://github.com/other/repo/actions/runs/900000")):
            evidence = copy.deepcopy(self.evidence)
            evidence["run"][key] = value
            self.assertFalse(score(self.request, self.row["chosen"], evidence)["available"])
        self.assertFalse(score({**self.request, "commit": "f" * 40}, self.row["chosen"], self.evidence)["available"])

    def test_source_missing_unknown_not_failed(self):
        for evidence in (None, {}, {"transport_status": 404}, {"transport_status": 200}):
            result = score(self.request, self.row["chosen"], evidence)
            self.assertFalse(result["available"])
            self.assertIsNone(result["score"])

    def test_answer_cannot_supply_its_own_authority(self):
        candidate = json.loads(self.row["chosen"])
        candidate["ci_observation"] = self.evidence
        self.assertFalse(score(self.request, json.dumps(candidate), None)["available"])

    def test_literal_native_boolean_and_pipeline(self):
        candidate = json.loads(self.row["chosen"])
        for value in ("true", 1, None, [], {}):
            candidate["passed"] = value
            self.assertEqual(score(self.request, json.dumps(candidate), self.evidence)["score"], 0)
        candidate = json.loads(self.row["chosen"])
        candidate["pipeline"] += "/"
        self.assertEqual(score(self.request, json.dumps(candidate), self.evidence)["score"], 0)

    def test_json_duplicate_nonfinite_and_depth(self):
        for text in ('{"passed":true,"passed":false}', '{"passed":NaN}', "[" * 2000 + "]" * 2000):
            with self.assertRaises(EvidenceError):
                strict_json(text)

    def test_url_variants_and_hostile_hosts(self):
        url = self.request["patch"]
        api = url.replace("https://github.com/", "https://api.github.com/repos/")
        self.assertEqual(run_identity(url), run_identity(api))
        for bad in (url + "?x=1", url + "#x", url + "/", url.replace("github.com", "github.com.evil"),
                    url.replace("github.com", "user@github.com"), url.replace("github.com", "github.com:443"),
                    url.replace("https:", "http:"), url.replace("cpv-fixture", "..")):
            with self.assertRaises(EvidenceError):
                run_identity(bad)

    def test_complete_job_inventory_deep_failure_visible(self):
        calls = []
        def fetch(page):
            calls.append(page)
            return {"total_count": 205, "jobs": [{"id": i, "conclusion": "failure" if i == 205 else "success"}
                    for i in range((page - 1) * 100 + 1, min(page * 100, 205) + 1)]}
        jobs = collect_jobs(fetch)
        self.assertEqual(calls, [1, 2, 3])
        self.assertEqual(len(jobs), 205)
        self.assertEqual(jobs[-1]["conclusion"], "failure")

    def test_incomplete_duplicate_changed_job_inventory_denied(self):
        for fetch in (lambda _: {"total_count": 101, "jobs": []},
                      lambda _: {"total_count": 2, "jobs": [{"id": 1}, {"id": 1}]},
                      lambda page: {"total_count": 101 if page == 1 else 102, "jobs": [{"id": 1}]}):
            with self.assertRaises(EvidenceError):
                collect_jobs(fetch)

    def test_invalid_types_never_become_source_truth(self):
        for key in ("id", "run_attempt", "repository", "head_sha", "html_url", "status", "conclusion"):
            for value in (None, True, [], {}, 1):
                evidence = copy.deepcopy(self.evidence)
                evidence["run"][key] = value
                result = score(self.request, self.row["chosen"], evidence)
                self.assertIn(result["score"], (None, 0, 1))

    def test_random_parser_inputs_bounded(self):
        rng = random.Random(451)
        for _ in range(200):
            text = bytes(rng.randrange(256) for _ in range(rng.randrange(80))).decode("latin1")
            try:
                strict_json(text)
            except EvidenceError:
                pass
        self.assertFalse(score(self.request, "x" * 20001, self.evidence)["score"])

    def test_literal_primary_renderer_ast(self):
        # Do not import trainer: its imports include model tooling.
        tree = ast.parse((HERE / "evidence/trainer.txt").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render")
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "primary-render", "exec"), namespace)
        for prompt, answer in ((" x\n", "\ttrue "), ("é\r\n", "{\"passed\":false}"), ("", "")):
            self.assertEqual(render(prompt, answer), namespace["render"](prompt, answer))


class DataGateTests(unittest.TestCase):
    def setUp(self):
        self.splits = {name: [fixture(name, i) for i in range(count)] for name, (count, _, _) in SPLITS.items()}

    def test_all_originals_and_normalized_disjointness(self):
        report = check_splits(self.splits)
        self.assertEqual([len(rows) for rows in self.splits.values()], [128, 32, 64])
        self.assertFalse(any(report["normalized_overlap"].values()))

    def test_overlap_detector_rejects_case_whitespace_variants(self):
        self.splits["audit"][0]["repository"] = "  " + self.splits["train"][0]["repository"].upper() + "  "
        with self.assertRaisesRegex(ValueError, "overlap"):
            check_splits(self.splits)

    def test_frozen_train_only_controls(self):
        controls = freeze_controls(self.splits["train"])
        before = copy.deepcopy(controls)
        for rows in self.splits.values():
            metrics = control_metrics(rows, controls)
            self.assertEqual(metrics["answer_only"]["accuracy"], 0.5)
            self.assertEqual(metrics["length"]["accuracy"], 0.5)
        self.assertEqual(controls, before)

    def test_corrupt_preference_refused(self):
        self.splits["train"][0]["chosen"] = self.splits["train"][0]["rejected"]
        with self.assertRaises(ValueError):
            check_splits(self.splits)


class RemoteGateTests(unittest.TestCase):
    def test_local_context_refuses_neural_work(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(RuntimeError):
            remote_only()

    def test_missing_or_false_caller_attestation_blocks(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            path = Path(folder) / "caller.json"
            with self.assertRaises(ValueError):
                caller_contract(path)
            path.write_text(json.dumps({"family": "CODE_PATCH_VERIFY", "trusted_ci_observation_in_literal_prompt": True}))
            with self.assertRaises(ValueError):
                caller_contract(path)

    def test_recipe_rejects_audit_selection_and_looser_gates(self):
        recipe = json.loads((HERE / "recipe.json").read_text())
        validate_recipe(recipe)
        for key, value in (("selection", {"split": "audit"}), ("quality", {"minimum_accuracy": 0.1}),
                           ("live_evidence_channel_verified", True), ("max_minutes", 999)):
            changed = {**recipe, key: value}
            with self.assertRaises(ValueError):
                validate_recipe(changed)

    def test_safetensors_stamp_preserves_payload_and_is_idempotent(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            path = Path(folder) / "control.safetensors"
            header = json.dumps({"__metadata__": {"original": "retained"}, "t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
            payload = struct.pack("<f", 1.0)
            path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
            terms = {"x-upstream-mit": "test upstream grant", "x-own-sand": "test own terms"}
            stamp_checkpoint(path, terms)
            once = path.read_bytes()
            stamp_checkpoint(path, terms)
            self.assertEqual(once, path.read_bytes())
            size = struct.unpack("<Q", once[:8])[0]
            self.assertEqual(once[8 + size:], payload)
            self.assertEqual(json.loads(once[8:8 + size])["__metadata__"]["original"], "retained")

    def test_torch_layout_zip_preserves_members_and_is_idempotent(self):
        # Container mechanics only. No Torch model, checkpoint or inference loaded.
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            path = Path(folder) / "control.pt"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("control/data.pkl", b"original opaque bytes")
                archive.writestr("control/data/0", b"1234")
            terms = {"x-upstream-mit": "test upstream grant", "x-own-sand": "test own terms"}
            stamp_checkpoint(path, terms)
            once = path.read_bytes()
            stamp_checkpoint(path, terms)
            self.assertEqual(once, path.read_bytes())
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(archive.read("control/data/0"), b"1234")
                self.assertEqual(json.loads(archive.read("control/cpv-licences.json")), terms)


if __name__ == "__main__":
    unittest.main()
