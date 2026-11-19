# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only wire, ready-report and restricted-lease failure controls."""
import copy
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from report import ENGINE, FOUNDATION, TOKENIZER, formatted, literal_reply, validate_ready
from serve import lease_modules

HERE = Path(__file__).resolve().parent


class OwnerControls(unittest.TestCase):
    def test_source_gate_blocks_neural_import_before_loading(self):
        environment = {**os.environ, "OWNER_SOURCE_ONLY": "1", "PYTHONPATH": os.environ.get("PYTHONPATH", "") + os.pathsep + str(HERE)}
        result = subprocess.run([sys.executable, "-c", "import torch"], env=environment, capture_output=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"neural imports forbidden", result.stderr)

    def test_upstream_notice_keeps_reviewed_own_terms_unchanged(self):
        original = (HERE / "support/qa-experiment/LICENSE").read_bytes()
        public = (HERE / "LICENSE").read_bytes()
        self.assertTrue(public.endswith(original))
        self.assertTrue(public.startswith(b"UPSTREAM PORTIONS AND ORIGINAL GRANTS"))
        self.assertEqual(__import__("hashlib").sha256(original).hexdigest(), "e6cf0cc83fbc51616590209d71ef1fa5fe46c130bef44e0d97d439ee91663447")
    def test_validation_acceptance_keeps_audit_closed_on_failure(self):
        from development_gate import accepted
        good = {"accuracy": .9, "per_label": {"Yes": .9, "No": .9}}
        self.assertTrue(accepted({"prose": good, "json": good}))
        self.assertFalse(accepted({"prose": good, "json": {"accuracy": .79, "per_label": {"Yes": .9, "No": .7}}}))
        self.assertFalse(accepted({"prose": good, "json": {"accuracy": .85, "per_label": {"Yes": .6, "No": .95}}}))
        with self.assertRaises(ValueError):
            accepted({"prose": good})

    def test_fresh_audit_excludes_all_previous_captures(self):
        boundary = json.loads((HERE / "data/audit-boundary.json").read_text())
        rows = [json.loads(line) for line in (HERE / "data/audit.jsonl").read_bytes().splitlines()]
        self.assertEqual(len({row["group_id"] for row in rows}), 32)
        self.assertEqual(len(boundary["excluded_prior_xml_files"]), 6144)
        self.assertFalse({row["source_path"] for row in rows} & set(boundary["excluded_prior_xml_files"]))
        self.assertFalse({row["group_id"] for row in rows} & set(boundary["exposed_audit_groups"]))
    def test_literal_bare_echo_not_normalized_standin(self):
        manifest = {"keccak256": "0x" + "a" * 64}
        value = {"foundation_hash": FOUNDATION, "tokenizer_hash": TOKENIZER, "adapter_hash": "a" * 64, "results": [{"score": 0}]}
        self.assertEqual(literal_reply(value, 1, manifest), [0])
        for key in ("foundation_hash", "tokenizer_hash", "adapter_hash"):
            bad = copy.deepcopy(value)
            bad[key] = "0x" + bad[key]
            with self.assertRaises(ValueError):
                literal_reply(bad, 1, manifest)

    def test_finite_bits_count_and_conflict_controls(self):
        base = {"foundation_hash": FOUNDATION, "tokenizer_hash": TOKENIZER, "adapter_hash": "a" * 64, "results": [{"score": 0}]}
        for result in ({"score": 0x7fc00000}, {"score": True}, {"score": 0, "error": "x"}, {}):
            with self.assertRaises(ValueError):
                literal_reply({**base, "results": [result]}, 1, {"keccak256": "a" * 64})
        with self.assertRaises(ValueError):
            literal_reply(base, 2, {"keccak256": "a" * 64})

    def test_json_format_uses_literal_raw_body(self):
        rows = [{"chosen": "Yes. A < B & C.", "rejected": "No. A < B & C."}]
        converted = formatted(rows, "json")
        self.assertEqual(converted[0]["chosen"], '{"answer":"Yes. A < B & C."}')
        self.assertEqual(json.loads(converted[0]["chosen"])["answer"], rows[0]["chosen"])
        self.assertEqual(rows[0]["chosen"], "Yes. A < B & C.")

    def report_fixture(self):
        # Explicit source-only contract fixture. Never emitted as a measured
        # remote report, package proof or registration input.
        metrics = {"accuracy": .9, "per_label": {"Yes": .9, "No": .9}, "untrained_accuracy": .5, "shortcut_accuracy": .5}
        return {"schema_version": 1, "status": "REGISTRATION_READY", "family": "RESEARCH_QUERY", "run_id": 1,
                "commit": "b" * 40, "serving_commit": "b" * 40, "package_sha256": "a" * 64, "package_keccak": "c" * 64,
                "foundation_sha256": FOUNDATION, "engine_sha256": ENGINE, "tokenizer_sha256": TOKENIZER,
                "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True,
                           "independent_groups": 32, "source_manifest_sha256": "d" * 64},
                "quality": {"passed": True, "audit_pairs": 64, "all_supported_formats_tested": True,
                            "formats": {"prose": dict(metrics), "json": dict(metrics)}},
                "runtime": {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
                            "negative_identity_controls": 3, "vectors": [{"tokens": [50281, 22, 50282], "score_bits": 0} for _ in range(8)]}}

    def test_ready_report_refuses_unexecuted_or_failed_gates(self):
        value = self.report_fixture()
        validate_ready(value)
        mutations = []
        bad = copy.deepcopy(value)
        bad["runtime"]["reference_engine_passed"] = False
        mutations.append(bad)
        bad = copy.deepcopy(value)
        bad["quality"]["formats"]["json"]["accuracy"] = .79
        mutations.append(bad)
        bad = copy.deepcopy(value)
        bad["source"]["audit_used_for_selection"] = True
        mutations.append(bad)
        bad = copy.deepcopy(value)
        bad["serving_commit"] = "a" * 40
        mutations.append(bad)
        bad = copy.deepcopy(value)
        bad["runtime"]["vectors"][0]["tokens"] = [True]
        mutations.append(bad)
        for bad in mutations:
            with self.assertRaises(ValueError):
                validate_ready(bad)

    def test_restricted_lease_contract_is_retained(self):
        lease, _ = lease_modules()
        public = "ssh-ed25519 " + __import__("base64").b64encode(bytes.fromhex("0000000b") + b"ssh-ed25519" + bytes.fromhex("00000020") + b"x" * 32).decode()
        row = {"family": "RESEARCH_QUERY", "owner": lease.DEFAULT_OWNER, "slot": 2, "forward_port": 48602,
               "runnable_url": "https://tga1.zkasuran.dev/_source-test/runnable", "keccak": "a" * 64,
               "sha256": "b" * 64, "existing_registration_id": 0,
               "candidate_proof": {"sha256": "c" * 64}, "runtime_proof": {"sha256": "c" * 64}, "source_proof": {"sha256": "c" * 64}}
        value = lease.lease_for(row, public, "source-test", "1", "d" * 40)
        self.assertIn('restrict,port-forwarding,permitlisten="127.0.0.1:48602"', value["authorized_keys_line"])
        self.assertIn('permitopen="registration-denied.invalid:1",command="/bin/false"', value["authorized_keys_line"])
        self.assertEqual({value[name] for name in ("candidate_proof_sha256", "runtime_proof_sha256", "source_proof_sha256")}, {"c" * 64})

    def test_serving_refuses_any_local_operation(self):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("GITHUB_")}
        result = subprocess.run([sys.executable, str(HERE / "serve.py"), "--prepare", "--root", str(HERE / "never-created")],
                                capture_output=True, timeout=5, env=environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((HERE / "never-created").exists())


if __name__ == "__main__":
    unittest.main()
