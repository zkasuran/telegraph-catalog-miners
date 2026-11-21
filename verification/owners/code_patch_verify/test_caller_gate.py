# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Synthetic proof-material controls, not CI labels or deployed caller evidence."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from caller_gate import validate_caller_contract
from prior_caller_gate import caller_contract as published_structural_gate

HERE = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class CallerGateTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=HERE)
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.path = self.root / "caller-contract.json"
        self.url = "https://github.com/cpv-fixture/proof/actions/runs/12345"
        self.observation = {"transport_status": 200, "observed_at": "2026-11-19T12:00:00Z", "run": {
            "id": 12345, "run_attempt": 1, "repository": {"full_name": "cpv-fixture/proof"},
            "head_sha": "a" * 40, "html_url": self.url, "status": "completed", "conclusion": "success"}}
        prompt = json.dumps({"patch": self.url, "ci_observation": self.observation})
        self.record = {"schema_version": 1, "family": "CODE_PATCH_VERIFY", "class": "judgment",
                       "caller_revision": "b" * 40, "caller_source_sha256": "0" * 64,
                       "literal_sample_prompt": prompt, "sample_sha256": digest(prompt.encode()),
                       "trusted_ci_observation_in_literal_prompt": True, "protected_rows_used": False,
                       "sample_origin": "independently_constructed_canonical_request"}

    def save(self):
        self.path.write_text(json.dumps(self.record))

    def bind_material(self):
        source = b"// Synthetic source-binding unit fixture. Not a deployed caller.\n"
        self.record["caller_source_sha256"] = digest(source)
        observation = json.dumps(self.observation).encode()
        trace = json.dumps({"caller_revision": self.record["caller_revision"], "caller_source_sha256": digest(source),
                            "source_observation_sha256": digest(observation), "prompt": self.record["literal_sample_prompt"],
                            "sample_origin": self.record["sample_origin"], "protected_rows_used": False}).encode()
        rows = {"caller_source": ("source/caller.go", source), "caller_licence": ("grants/LICENSE.txt", b"Synthetic test grant, not permission to use a real third-party caller.\n"),
                "literal_trace": ("trace/capture.json", trace), "ci_observation": ("observations/ci.json", observation)}
        self.record["evidence_files"] = {}
        for role, (name, raw) in rows.items():
            relative = "caller-proof/" + name
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            self.record["evidence_files"][role] = {"path": relative, "sha256": digest(raw)}
        self.save()

    def test_published_gate_accepts_unbound_claims_but_hardened_gate_refuses(self):
        self.save()
        self.assertEqual(published_structural_gate(self.path)["caller_source_sha256"], "0" * 64)
        with self.assertRaisesRegex(ValueError, "flags are not evidence"):
            validate_caller_contract(self.path)

    def test_material_consistency_is_not_deployment_authentication(self):
        self.bind_material()
        result = validate_caller_contract(self.path)
        self.assertEqual(len(result["byte_bound_material_roles"]), 4)
        self.assertEqual(result["deployment_authentication"], "not_established_by_this_validator")

    def test_source_byte_tampering_refused(self):
        self.bind_material()
        (self.root / "caller-proof/source/caller.go").write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_caller_contract(self.path)

    def test_trace_prompt_change_refused_even_with_refreshed_file_hash(self):
        self.bind_material()
        path = self.root / "caller-proof/trace/capture.json"
        trace = json.loads(path.read_text())
        trace["prompt"] = "different literal input"
        path.write_text(json.dumps(trace))
        self.record["evidence_files"]["literal_trace"]["sha256"] = digest(path.read_bytes())
        self.save()
        with self.assertRaisesRegex(ValueError, "literal trace"):
            validate_caller_contract(self.path)

    def test_missing_material_refused(self):
        self.bind_material()
        (self.root / "caller-proof/grants/LICENSE.txt").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_caller_contract(self.path)

    def test_declared_source_identity_mismatch_refused(self):
        self.bind_material()
        self.record["caller_source_sha256"] = "0" * 64
        self.save()
        with self.assertRaisesRegex(ValueError, "source identity"):
            validate_caller_contract(self.path)

    def test_trace_numeric_false_is_not_a_native_false_boundary(self):
        self.bind_material()
        path = self.root / "caller-proof/trace/capture.json"
        trace = json.loads(path.read_text())
        trace["protected_rows_used"] = 0
        path.write_text(json.dumps(trace))
        self.record["evidence_files"]["literal_trace"]["sha256"] = digest(path.read_bytes())
        self.save()
        with self.assertRaisesRegex(ValueError, "native false"):
            validate_caller_contract(self.path)

    def test_literal_observation_float_run_identity_refused(self):
        self.bind_material()
        sample = json.loads(self.record["literal_sample_prompt"])
        sample["ci_observation"]["run"]["id"] = float(self.observation["run"]["id"])
        prompt = json.dumps(sample)
        self.record["literal_sample_prompt"] = prompt
        self.record["sample_sha256"] = digest(prompt.encode())
        path = self.root / "caller-proof/trace/capture.json"
        trace = json.loads(path.read_text())
        trace["prompt"] = prompt
        path.write_text(json.dumps(trace))
        self.record["evidence_files"]["literal_trace"]["sha256"] = digest(path.read_bytes())
        self.save()
        with self.assertRaisesRegex(ValueError, "literal sample"):
            validate_caller_contract(self.path)

    def test_symlink_refused_before_read(self):
        self.bind_material()
        path = self.root / "caller-proof/source/caller.go"
        path.unlink()
        path.symlink_to(self.root / "caller-proof/trace/capture.json")
        with self.assertRaisesRegex(ValueError, "symlink"):
            validate_caller_contract(self.path)

    def test_path_escape_and_private_names_refused(self):
        self.bind_material()
        for value in ("../outside.go", "caller-proof/source/.env", "caller-proof/source/auth.json"):
            self.record["evidence_files"]["caller_source"]["path"] = value
            self.save()
            with self.assertRaises(ValueError):
                validate_caller_contract(self.path)

    def test_non_object_and_duplicate_key_contracts_refused(self):
        for value in ("[]", '{"schema_version":1,"schema_version":1}'):
            self.path.write_text(value)
            with self.assertRaises(ValueError):
                validate_caller_contract(self.path)


if __name__ == "__main__":
    unittest.main()
