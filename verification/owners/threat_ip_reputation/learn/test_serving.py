# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only serving contracts, mechanical fixtures only, never audit reports."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import contracts
import serving

HERE = Path(__file__).resolve().parent


def fixture():
    lock = json.loads((HERE / "candidate-lock.json").read_text())
    return {"schema_version": 1, "family": contracts.FAMILY, "status": "REGISTRATION_READY", "fit_on_audit": False,
            "run_id": 1, "commit": "0" * 40, "package_sha256": lock["package_sha256"], "package_keccak": lock["package_keccak"], "package_bytes": lock["package_bytes"],
            "foundation_sha256": contracts.FOUNDATION["foundation_lora_fp32.onnx"], "engine_sha256": contracts.FOUNDATION["foundation_engine.wasm"], "tokenizer_sha256": contracts.FOUNDATION["tokenizer.json"],
            "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True, "independent_groups": 32, "source_manifest_sha256": "a" * 64},
            "quality": {"passed": True, "audit_pairs": 64, "all_supported_formats_tested": True,
                        "formats": {fmt: {"accuracy": 1.0, "per_label": {label: 1.0 for label in contracts.LABELS}, "untrained_accuracy": 0.5, "shortcut_accuracy": 0.5} for fmt in contracts.FORMATS}},
            "runtime": {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True, "negative_identity_controls": 3,
                        "vectors": [{"tokens": [1], "score_bits": 0} for _ in range(8)]}, "fixture_scope": "mechanical schema controls, not actual audit/registration evidence"}


class ServingTests(unittest.TestCase):
    def test_pure_contract_primary_and_native_shortcut_refusal(self):
        value = fixture()
        self.assertEqual(serving.ready_report(json.dumps(value).encode())["family"], contracts.FAMILY)
        value["status"] = "CANDIDATE_FROZEN_AUDIT_REQUIRED"
        with self.assertRaises(ValueError):
            serving.ready_report(json.dumps(value).encode())

    def test_source_quality_and_nonfinite_refusals(self):
        for mutate in (lambda value: value["source"].update(independent_groups=31),
                       lambda value: value["source"].update(audit_used_for_selection=True),
                       lambda value: value["quality"]["formats"]["prose"].update(accuracy=0.75),
                       lambda value: value["runtime"]["vectors"][0].update(score_bits=0x7fc00000)):
            value = fixture()
            mutate(value)
            with self.assertRaises(ValueError):
                serving.ready_report(json.dumps(value).encode())

    def test_exact_reservation_and_combined_proof_hashes(self):
        value = fixture()
        reservation = {"family": contracts.FAMILY, "package_keccak": value["package_keccak"], "slot": 6, "forward_port": 48606,
                       "ssh_host": serving.lease_common.HOST, "ssh_user": serving.lease_common.SSH_USER,
                       "runnable_url": "https://tga1.zkasuran.dev/_s2o_threat_ip_reputation_" + value["package_keccak"][:8] + "/runnable"}
        row = serving.lease_row(value, reservation, "a" * 64)
        for kind in ("candidate", "runtime", "source"):
            self.assertEqual(row[kind + "_proof"]["sha256"], "a" * 64)
        reservation["forward_port"] = 48605
        with self.assertRaises(ValueError):
            serving.lease_row(value, reservation, "a" * 64)

    def test_literal_restricted_ssh_argv_without_execution(self):
        command = serving.ssh_command(Path("mechanical-root"), {"forward_port": 48606}, 12345)
        self.assertIn("127.0.0.1:48606:127.0.0.1:12345", command)
        for control in ("IdentityAgent=none", "StrictHostKeyChecking=yes", "ForwardAgent=no", "PermitLocalCommand=no", "ForwardX11=no"):
            self.assertIn(control, command)

    def test_real_serving_cli_refuses_local_work_before_key_or_model(self):
        process = subprocess.run([sys.executable, "-W", "error", str(HERE / "serving.py"), "--mode", "prepare", "--root", "unused"],
                                 capture_output=True, text=True, timeout=10, env={**os.environ, "GITHUB_ACTIONS": "false"})
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("model work runs only", process.stderr)
        self.assertNotIn("No module named", process.stderr)


if __name__ == "__main__":
    unittest.main()
