# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Synthetic protocol fixtures only. No fixture is a measured model report."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from common import DEFAULT_OWNER, FOUNDATION
from contracts import validate_ready
from groups import components
from proxy import input_root, validate_request, validate_response
from serving import public_asset
from remote import read_split
from verify import config_gate, source_proof

HERE = Path(__file__).resolve().parent


def synthetic_ready_fixture():
    # These numbers exercise the parser's thresholds, not neural quality.
    return {"schema_version": 1, "status": "REGISTRATION_READY", "family": "EVENT_OUTCOME_RESOLUTION",
            "run_id": 123, "commit": "a"*40, "serving_commit": "a"*40, "package_sha256": "b"*64,
            "package_keccak": "c"*64, "owner": DEFAULT_OWNER,
            "foundation_sha256": FOUNDATION["foundation_lora_fp32.onnx"], "engine_sha256": FOUNDATION["foundation_engine.wasm"],
            "tokenizer_sha256": FOUNDATION["tokenizer.json"],
            "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True,
                       "source_manifest_sha256": "d"*64, "independent_groups": 32},
            "quality": {"passed": True, "audit_pairs": 64, "all_supported_formats_tested": True,
                        "formats": {name: {"accuracy": 0.8, "per_label": {label: 0.7 for label in ("yes", "no", "unresolved")},
                                           "untrained_accuracy": 0.75, "shortcut_accuracy": 0.7} for name in ("json", "prose")}},
            "runtime": {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
                        "negative_identity_controls": 3, "vectors": [{"tokens": [50281, 22, 50282], "score_bits": 0} for _ in range(8)]}}


class Contracts(unittest.TestCase):
    def test_complete_synthetic_contract(self):
        self.assertIsInstance(validate_ready(synthetic_ready_fixture()), dict)

    def test_actual_source_group_count(self):
        proof = source_proof()
        self.assertEqual(proof["independent_groups"], 39)
        self.assertEqual(proof["audit_pairs"], 384)
        self.assertEqual(proof["normalized_overlap"], 0)

    def test_events_are_not_independent_people(self):
        rows = [{"event_id": str(i), "people": ["shared"]} for i in range(64)]
        self.assertEqual(components(rows), 1)
        self.assertEqual(components([{**r, "people": [str(i)]} for i, r in enumerate(rows)]), 64)

    def test_identity_and_complete_audit_failure_controls(self):
        fixture = synthetic_ready_fixture()
        for path, value in [("source.independent_groups", 31), ("source.audit_used_for_selection", True),
                            ("quality.audit_pairs", 63), ("quality.formats.prose.accuracy", 0.79),
                            ("quality.formats.json.per_label.no", 0.69), ("runtime.negative_identity_controls", 2),
                            ("runtime.repeat_bits_identical", False), ("foundation_sha256", "0"*64),
                            ("serving_commit", "e"*40), ("package_keccak", "0x" + "c"*64)]:
            bad = copy.deepcopy(fixture)
            target = bad
            keys = path.split(".")
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            with self.assertRaises(ValueError):
                validate_ready(bad)

    def test_nonfinite_runtime_and_native_score_refusals(self):
        for value in (True, 1.0, -1, 0x7f800000, 0x7fc00000, 1 << 32):
            bad = synthetic_ready_fixture()
            bad["runtime"]["vectors"][0]["score_bits"] = value
            with self.assertRaises(ValueError):
                validate_ready(bad)

    def test_literal_reference_wire_not_normalized_proxy(self):
        row = {"keccak": "c"*64}
        request = {"batch": [{"tokens": [50281, 22, 50282]}], "foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
                   "tokenizer_hash": FOUNDATION["tokenizer.json"], "adapter_hash": row["keccak"]}
        request["input_root"] = input_root(request["batch"])
        positive = validate_request(request, row)
        response = {**{k: v for k, v in positive.items() if k != "batch"}, "results": [{"score": 0x3f800000}]}
        self.assertEqual(validate_response(response, positive, row)["results"], [{"score": 0x3f800000}])
        for key in ("foundation_hash", "tokenizer_hash", "adapter_hash"):
            for value in (request[key].upper(), "0x" + request[key], "0"*64):
                with self.assertRaises(ValueError):
                    validate_request({**request, key: value}, row)
                with self.assertRaises(ValueError):
                    validate_response({**response, key: value}, positive, row)

    def test_owned_immutable_serving_assets_only(self):
        valid = "https://raw.githubusercontent.com/zkasuran/telegraph-catalog-miners/" + "a"*40 + "/verification/owners/event_outcome_resolution/releases/model.safetensors"
        self.assertEqual(public_asset(valid), valid)
        for bad in (valid + "?key=anything", valid.replace("releases/model", "../model"), valid.replace("event_outcome_resolution", "other"), "http://localhost/x"):
            with self.assertRaises(ValueError):
                public_asset(bad)

    def test_all_neural_entrypoints_refuse_local(self):
        for name in ("remote.py", "runtime.py", "serving.py", "report.py"):
            result = subprocess.run([sys.executable, str(HERE / name), "--help"], capture_output=True, timeout=10,
                                    env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"exact owned public CPU workflow", result.stderr)
            self.assertNotIn(b"torch", result.stderr)

    def test_workflow_full_standalone_modes_and_artifact_paths(self):
        config_gate()
        workflow = json.loads((HERE.parents[2] / ".github/workflows/evaluator-owner-event_outcome_resolution.yml").read_text())
        self.assertEqual(set(workflow["jobs"]), {"train", "audit", "report", "serve"})
        upload = next(s for s in workflow["jobs"]["train"]["steps"] if s.get("with", {}).get("name") == "event-candidate")
        self.assertTrue(all(p.startswith("${{ runner.temp }}/event-outcome-training/") for p in upload["with"]["path"].splitlines()))
        serving = next(s for s in workflow["jobs"]["serve"]["steps"] if s.get("with", {}).get("name") == "event-serving-lease")
        self.assertEqual(serving["with"]["path"], "${{ runner.temp }}/event-serving/lease.json")

    def test_actual_upstream_pair_collator_margin_contract(self):
        manifest = json.loads((HERE / "frozen/frozen.json").read_text())
        rows = read_split("train", manifest)
        self.assertTrue(all(type(r["margin"]) is float and r["margin"] == 0.0 for r in rows))
        primary = (HERE / "sources/kit-float_runtime_train_adapter.py").read_text()
        self.assertIn('[p["margin"] for p in batch]', primary)


if __name__ == "__main__":
    unittest.main()
