# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only tests. Fixtures never load a model, signer or private pool."""
import base64
import copy
import io
import json
import os
import struct
import threading
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import contracts
import remote
import serve
from common import lease_for, validate_lease
from contracts import (
    FAMILY,
    FORMATS,
    FOUNDATION,
    FROZEN,
    OWNER,
    file_sha,
    finite_bits,
    package_parts,
    recover_package,
    sha,
    strict_json,
    validate_ready,
    verify_recovery,
)
from proxy import BoundedServer, identity, validate_request, validate_response
from source_data import group_id, inventory, pair, shortcuts


def ready():
    return {"schema_version": 1, "status": "REGISTRATION_READY", "family": FAMILY, "run_id": 12345,
            "commit": "a"*40, "serving_commit": "a"*40, "package_sha256": "b"*64, "package_keccak": "c"*64,
            "foundation_sha256": FOUNDATION["foundation_lora_fp32.onnx"], "engine_sha256": FOUNDATION["foundation_engine.wasm"],
            "tokenizer_sha256": FOUNDATION["tokenizer.json"],
            "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True,
                       "independent_groups": 256, "source_manifest_sha256": "d"*64},
            "quality": {"passed": True, "audit_pairs": 256, "all_supported_formats_tested": True,
                        "formats": {name: {"accuracy": .90, "per_label": {"Positive": .9, "Negative": .9},
                                           "untrained_accuracy": .5, "shortcut_accuracy": .5} for name in FORMATS}},
            "runtime": {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
                        "negative_identity_controls": 3, "vectors": [{"tokens": ids, "score_bits": 0x3f800000} for ids in remote.fixed_vectors()]}}


class ContractTests(unittest.TestCase):
    def test_full_inventory_checks_every_original_domain_record(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for domain, filename in (("amazon", "amazon_cells"), ("imdb", "imdb"), ("yelp", "yelp")):
                content = "".join(f"Source-only synthetic {domain} fixture {i}\t{i % 2}\n" for i in range(1000))
                archive.writestr("sentiment labelled sentences/" + filename + "_labelled.txt", content)
        rows, counts = inventory(io.BytesIO(buffer.getvalue()))
        self.assertEqual(len(rows), 3000)
        self.assertEqual(counts, {"amazon": 1000, "imdb": 1000, "yelp": 1000})

    def test_public_export_keeps_frozen_byte_identity_and_original_grants(self):
        path = contracts.HERE / "model/SENTIMENT_ANALYSIS-recovery.safetensors"
        self.assertEqual(file_sha(path), FROZEN["recovered_sha256"])
        header, payload = package_parts(path.read_bytes())
        metadata = header["__metadata__"]
        self.assertIn("LicenseRef-zkasuran-SAND-1.0", metadata["x-licence"])
        self.assertIn("Apache License", metadata["x-upstream-apache-2.0"])
        self.assertIn("Permission is hereby granted", metadata["x-upstream-mit"])
        self.assertEqual(metadata["intent"], FAMILY)
        self.assertEqual(metadata["class"], "judgment")
        self.assertTrue(payload)
    def test_lease_binds_all_three_proofs_to_actual_combined_report_hash(self):
        row = {"family": FAMILY, "owner": OWNER, "slot": 1, "forward_port": 48601,
               "runnable_url": "https://tga1.zkasuran.dev/_s2owner/runnable", "keccak": "a"*64,
               "sha256": "b"*64, "existing_registration_id": 0}
        for kind in ("candidate", "runtime", "source"):
            row[kind + "_proof"] = {"sha256": "c"*64}
        public = struct.pack(">I", 11) + b"ssh-ed25519" + struct.pack(">I", 32) + b"p"*32
        key = "ssh-ed25519 " + base64.b64encode(public).decode()
        lease = lease_for(row, key, "source-only", "12345", "a"*40)
        validate_lease(lease, row)
        self.assertEqual(lease["owner"], OWNER)
        for kind in ("candidate", "runtime", "source"):
            self.assertEqual(lease[kind + "_proof_sha256"], "c"*64)
    def test_exact_ready_contract_accepts_complete_actual_shape(self):
        validate_ready(ready())

    def test_missing_format_and_wrong_run_assets_hashes_refused(self):
        for key, value in (("run_id", "12345"), ("family", "OTHER"), ("package_keccak", "0x" + "c"*64),
                           ("engine_sha256", "0"*64), ("serving_commit", "b"*40)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_ready({**ready(), key: value})
        report = ready()
        report["quality"]["formats"].pop("bare_label")
        with self.assertRaises(ValueError):
            validate_ready(report)

    def test_source_overlap_selection_grants_and_group_count_refused(self):
        for key, value in (("normalized_overlap", 1), ("audit_used_for_selection", True),
                           ("grants_verified", False), ("independent_groups", 31)):
            report = ready()
            report["source"][key] = value
            with self.assertRaises(ValueError):
                validate_ready(report)

    def test_nonfinite_missing_label_and_threshold_failures_refused(self):
        for change in ({"accuracy": .79}, {"accuracy": float("nan")}, {"shortcut_accuracy": .85},
                       {"untrained_accuracy": .88}, {"per_label": {"Positive": .9}},
                       {"per_label": {"Positive": .9, "Negative": .69}}):
            report = ready()
            report["quality"]["formats"]["prose"].update(change)
            with self.assertRaises(ValueError):
                validate_ready(report)

    def test_runtime_parser_identity_repeats_and_real_vectors_required(self):
        for key, value in (("full_parser_passed", False), ("reference_engine_passed", False),
                           ("repeat_bits_identical", False), ("negative_identity_controls", 2), ("vectors", [])):
            report = ready()
            report["runtime"][key] = value
            with self.assertRaises(ValueError):
                validate_ready(report)
        for bits in (True, -1, 1 << 32, 0x7fc00000, 0x7f800000):
            with self.assertRaises(ValueError):
                finite_bits(bits)

    def test_duplicate_json_keys_and_bad_containers_refused(self):
        for raw in (b'{"score":1,"score":2}', b'{"score":NaN}'):
            with self.assertRaises(ValueError):
                strict_json(raw)
        for raw in (b"", b"bad", b"\0"*12):
            with self.assertRaises(ValueError):
                package_parts(raw)

    def test_remote_guards_precede_all_neural_or_key_operations(self):
        for environment in ({}, {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": contracts.REPO,
                                  "GITHUB_EVENT_NAME": "workflow_dispatch", "OWNER_SOURCE_ONLY": "1"}):
            with patch.dict(os.environ, environment, clear=True), self.assertRaises(RuntimeError):
                remote.audit(Path("source-only-fixture"))
            with patch.dict(os.environ, environment, clear=True), self.assertRaises(RuntimeError):
                serve.prepare(Path("source-only-fixture"), None, None, None, None, None, None)

    def fixture_package(self):
        metadata = {"intent": FAMILY, "class": "judgment", "x-licence": "LicenseRef-zkasuran-SAND-1.0",
                    "x-upstream-notice": "retained", "x-upstream-apache-2.0": "Apache-2.0 grant", "x-upstream-mit": "MIT grant"}
        header = {"__metadata__": metadata, "fixture": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
        encoded = json.dumps(header).encode()
        return struct.pack("<Q", len(encoded)) + encoded + struct.pack("<f", 1.0)

    def test_recovery_preserves_model_fields_grants_descriptors_and_payload(self):
        original = self.fixture_package()
        with patch.dict(FROZEN, {"origin_sha256": sha(original)}):
            recovered = recover_package(original, FROZEN["recovery_id"])
            before, payload = package_parts(original)
            after, new_payload = package_parts(recovered)
            self.assertEqual(payload, new_payload)
            self.assertEqual(before["fixture"], after["fixture"])
            self.assertEqual(verify_recovery(original, recovered), sha(payload))
            self.assertNotEqual(sha(original), sha(recovered))
            with self.assertRaises(ValueError):
                verify_recovery(original, recovered[:-4] + struct.pack("<f", 2.0))
            changed = copy.deepcopy(after)
            changed["__metadata__"]["class"] = "hybrid"
            encoded = json.dumps(changed).encode()
            with self.assertRaises(ValueError):
                verify_recovery(original, struct.pack("<Q", len(encoded)) + encoded + new_payload)

    def test_wrong_origin_hash_or_recovery_identity_refused(self):
        with self.assertRaises(ValueError):
            recover_package(self.fixture_package(), FROZEN["recovery_id"])
        with self.assertRaises(ValueError):
            recover_package(self.fixture_package(), "different")

    def test_immutable_public_url_refuses_branch_foreign_or_credentials(self):
        for url in ("http://example.com", FROZEN["origin_url"].replace("37e3c48cfc71e182adf05012a493325a367b0fe2", "master"),
                    FROZEN["origin_url"] + "?token=fixture", FROZEN["origin_url"].replace("/artifacts/", "/../")):
            with self.assertRaises(ValueError):
                contracts.immutable_url(url)

    def test_frozen_baseline_coefficients_bind_training_only(self):
        path = contracts.HERE / "frozen-baseline.json"
        coefficient = strict_json(path.read_bytes())
        self.assertEqual(file_sha(path), FROZEN["baseline_sha256"])
        self.assertEqual(coefficient["train_sha256"], FROZEN["splits"]["train"]["sha256"])
        rows = []
        for label in ("Positive", "Negative"):
            text = "Source-only bounded fixture text."
            rows.append(pair({"text": text, "label": label, "group_id": group_id(text), "source_id": label, "domain": "fixture"}))
        self.assertEqual(shortcuts(rows, coefficient)["length"], .5)
        self.assertEqual(shortcuts(rows, coefficient)["answer_only"], .5)

    def test_normalized_group_and_all_three_response_views(self):
        self.assertEqual(group_id(" A  café "), group_id("a cafe\u0301"))
        row = {"text": "Source-only fixture", "label": "Positive", "group_id": "a"*64, "source_id": "fixture", "domain": "fixture"}
        views = [pair(row, style) for style in range(3)]
        self.assertEqual([r["chosen"] for r in views][-1], "Positive")
        self.assertTrue(all(r["chosen"] != r["rejected"] for r in views))
        with self.assertRaises(ValueError):
            pair(row, 3)


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.row = {"keccak": "a"*64, "runnable_url": "https://tga1.zkasuran.dev/_s2owner/runnable"}
        row = self.row

        class FixtureBackend:
            calls = 0
            def score(self, request):
                self.calls += 1
                # The upstream private executor may prefix its identity. Public output must not.
                return {**identity(row), "adapter_hash": "0x" + row["keccak"], "input_root": request.get("input_root", ""),
                        "results": [{"score": 0x3f800000} for _ in request["batch"]]}
        self.backend = FixtureBackend()
        self.server = BoundedServer(("127.0.0.1", 0), self.row, self.backend)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_primary_http_emits_literal_bare_hash_and_finite_bits(self):
        import http.client

        from common import json_bytes
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            request = {"batch": [{"tokens": [50281, 22, 23, 50282]}], **identity(self.row)}
            connection.request("POST", "/runnable", json_bytes(request), {"Content-Type": "application/json"})
            response = connection.getresponse()
            value = strict_json(response.read())
            self.assertEqual(response.status, 200)
            self.assertEqual(value["adapter_hash"], self.row["keccak"])
            self.assertFalse(value["adapter_hash"].startswith("0x"))
            self.assertEqual(value["results"], [{"score": 0x3f800000}])
        finally:
            connection.close()

    def test_actual_negative_identity_and_token_controls_before_compute(self):
        self.assertEqual(remote.identity_controls(self.server, self.row), 3)
        self.assertEqual(self.backend.calls, 0)

    def test_size_count_type_and_nonfinite_controls_refuse(self):
        for tokens in ([], [-1], [True], [50368], [1]*513):
            with self.assertRaises(ValueError):
                validate_request({"batch": [{"tokens": tokens}], **identity(self.row)}, self.row)
        request = {"batch": [{"tokens": [1]}], **identity(self.row)}
        for results in ([], [{"score": 0x7fc00000}], [{"score": 1}, {"score": 2}], [{}]):
            with self.assertRaises(ValueError):
                validate_response({"results": results, **identity(self.row)}, request, self.row)


if __name__ == "__main__":
    unittest.main()
