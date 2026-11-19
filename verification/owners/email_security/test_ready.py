# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Artificial contract fixtures only. These do not emit a real ready report."""
import copy
import unittest

import ready
import serve
import wire_common
import wire_proxy
from runtime import HASHES


def artificial_report():
    # Local parser fixture, not neural results, source truth or a candidate.
    return {"schema_version": 1, "status": "REGISTRATION_READY", "family": "EMAIL_SECURITY", "run_id": 123,
            "commit": "a" * 40, "serving_commit": "a" * 40, "package_sha256": "b" * 64, "package_keccak": "c" * 64,
            "foundation_sha256": HASHES["foundation_lora_fp32.onnx"], "engine_sha256": HASHES["foundation_engine.wasm"],
            "tokenizer_sha256": HASHES["tokenizer.json"],
            "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True,
                       "source_manifest_sha256": "d" * 64, "independent_groups": 64},
            "quality": {"passed": True, "audit_pairs": 128, "all_supported_formats_tested": True,
                        "formats": {"dns_json": {"accuracy": .90, "per_label": {"name": .90, "type": .90,
                                "lookup_ok_true": .90, "lookup_ok_false": .90}, "untrained_accuracy": .80, "shortcut_accuracy": .75}}},
            "runtime": {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
                        "negative_identity_controls": 3,
                        "vectors": [{"tokens": [50281, 10 + i, 50282], "score_bits": 0x3f800000} for i in range(8)]}}


class ReadyContractTests(unittest.TestCase):
    def test_artificial_contract_parses(self):
        ready.verify_ready(artificial_report())

    def test_actual_minimum_gates_cannot_be_lowered(self):
        for key, value in (("accuracy", .79), ("untrained_accuracy", .88), ("shortcut_accuracy", .85)):
            report = artificial_report()
            report["quality"]["formats"]["dns_json"][key] = value
            with self.assertRaises(ValueError):
                ready.verify_ready(report)
        report = artificial_report()
        report["quality"]["formats"]["dns_json"]["per_label"]["lookup_ok_false"] = .69
        with self.assertRaises(ValueError):
            ready.verify_ready(report)
        self.assertFalse(ready.quality_gate(.85, {"true": .85, "false": .85}, .80, .75))

    def test_report_omissions_and_identity_failures(self):
        mutations = []
        for key in ("full_parser_passed", "reference_engine_passed", "repeat_bits_identical"):
            report = artificial_report()
            report["runtime"][key] = False
            mutations.append(report)
        report = artificial_report()
        report["runtime"]["vectors"] = report["runtime"]["vectors"][:4]
        mutations.append(report)
        report = artificial_report()
        report["source"]["audit_used_for_selection"] = True
        mutations.append(report)
        report = artificial_report()
        report["source"]["independent_groups"] = 31
        mutations.append(report)
        report = artificial_report()
        report["runtime"]["vectors"][0]["score_bits"] = 0x7fc00000
        mutations.append(report)
        report = artificial_report()
        report["package_keccak"] = "0x" + report["package_keccak"]
        mutations.append(report)
        for report in mutations:
            with self.assertRaises(ValueError):
                ready.verify_ready(report)

    def test_literal_bare_serving_echo(self):
        report = artificial_report()
        vectors = report["runtime"]["vectors"]
        batch = [{"tokens": vector["tokens"]} for vector in vectors]
        reply = {"foundation_hash": report["foundation_sha256"], "adapter_hash": report["package_keccak"],
                 "tokenizer_hash": report["tokenizer_sha256"], "input_root": wire_proxy.input_root(batch),
                 "results": [{"score": vector["score_bits"]} for vector in vectors]}
        serve.check_vector_reply(reply, report, vectors)
        for key in ("foundation_hash", "adapter_hash", "tokenizer_hash"):
            with self.assertRaises(ValueError):
                serve.check_vector_reply({**reply, key: "0x" + reply[key]}, report, vectors)

    def test_forward_lease_binds_combined_report(self):
        import base64
        import struct
        report = artificial_report()
        row = serve.row_from_report(report, "e" * 64, 2, 48602, "https://tga1.zkasuran.dev/_emailtest/runnable")
        # Public-key wire fixture only, no private key or SSH operation.
        blob = struct.pack(">I", 11) + b"ssh-ed25519" + struct.pack(">I", 32) + b"k" * 32
        key = "ssh-ed25519 " + base64.b64encode(blob).decode()
        lease = wire_common.lease_for(row, key, "email-test", "123", "a" * 40)
        self.assertEqual(lease["candidate_proof_sha256"], "e" * 64)
        self.assertEqual(lease["runtime_proof_sha256"], "e" * 64)
        self.assertEqual(lease["source_proof_sha256"], "e" * 64)
        self.assertIn('permitlisten="127.0.0.1:48602"', lease["authorized_keys_line"])
        self.assertIn('command="/bin/false"', lease["authorized_keys_line"])
        changed = copy.deepcopy(lease)
        changed["adapter_hash"] = "f" * 64
        with self.assertRaises(ValueError):
            wire_common.validate_lease(changed, row)


if __name__ == "__main__":
    unittest.main()
