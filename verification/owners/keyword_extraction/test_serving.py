# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Synthetic lease/wire and real local HTTP failure controls, never neural evidence."""
import base64
import copy
import http.client
import json
import os
import struct
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import lease_common
import serve
from heldout_freeze import verify_freeze, wire
from serving_contract import check_run, serving_row
from source import sha
from supervise import timeout_for
from test_ready_contract import synthetic_report
from token_proxy import BoundedServer, identity, input_root, validate_request


def settings():
    base = "https://raw.githubusercontent.com/zkasuran/telegraph-catalog-miners/" + "a" * 40 + "/verification/owners/keyword_extraction/"
    return {"package_url": base + "results/KEYWORD_EXTRACTION.safetensors", "report_url": base + "results/registration-ready.json",
            "report_sha256": "d" * 64, "slot": 2, "forward_port": 48602,
            "runnable_url": "https://tga1.zkasuran.dev/_s2o_keyword_extraction_" + "c" * 8 + "/runnable"}


def synthetic_row():
    report = synthetic_report()
    report["runtime"]["executor_sha256"] = "e" * 64
    return serving_row(settings(), report, "a" * 40)


class ServingTests(unittest.TestCase):
    def test_precision_donor_attribution_and_cap_are_not_bypassed(self):
        from gates import check_splits
        from test_keyword import fixture
        splits = {s: [fixture(doi=s, author=s, group=s)] for s in ("train", "validation", "test")}
        for s, rows in splits.items():
            rows[0]["source"]["text"] = s
            rows[0]["chosen"], rows[0]["rejected"] = s + " alpha", s + " beta"
        donor = {**splits["train"][0], "off_topic_donor": True, "donor_source": splits["train"][0]["source"]}
        bad = copy.deepcopy(splits)
        bad["train"] = [donor]
        with self.assertRaisesRegex(ValueError, "exceeds 25%"):
            check_splits(bad)
        bad["train"] = [splits["train"][0]] * 3 + [donor]
        self.assertTrue(check_splits(bad))
        bad["train"][-1] = {**donor, "donor_source": splits["test"][0]["source"]}
        with self.assertRaisesRegex(ValueError, "overlap"):
            check_splits(bad)

    def test_heldout_source_freeze_refuses_changed_bytes(self):
        # Synthetic source-count mechanics only, never training or audit material.
        splits = {name: [{"group_id": "synthetic-" + str(i // 2), "chosen": "test alpha", "rejected": "test beta"} for i in range(n)]
                  for name, n in (("validation", 16), ("test", 64))}
        manifest = {"schema_version": 1, "family": "KEYWORD_EXTRACTION", "status": "HELDOUT_SOURCE_FROZEN",
                    "files": {name: {"sha256": sha(wire(rows)), "pairs": len(rows), "groups": len({r["group_id"] for r in rows})} for name, rows in splits.items()}}
        self.assertTrue(verify_freeze(splits, manifest))
        changed = copy.deepcopy(splits)
        changed["test"][0]["chosen"] = "changed synthetic bytes"
        with self.assertRaisesRegex(ValueError, "source bytes/counts changed"):
            verify_freeze(changed, manifest)
        with self.assertRaises(ValueError):
            verify_freeze(splits, {**manifest, "schema_version": True})

    def test_exact_report_reservation_and_source_commit_bindings(self):
        report = synthetic_report()
        report["runtime"]["executor_sha256"] = "e" * 64
        row = serving_row(settings(), report, "a" * 40)
        self.assertEqual(row["candidate_proof"], row["source_proof"])
        self.assertEqual(row["source_proof"], row["runtime_proof"])
        for changed in ({"forward_port": 48603}, {"slot": True}, {"runnable_url": "https://example.com/runnable"},
                        {"package_url": settings()["package_url"].replace("keyword_extraction", "other_family")}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                serving_row({**settings(), **changed}, report, "a" * 40)
        with self.assertRaises(ValueError):
            serving_row(settings(), report, "b" * 40)
        report["quality"]["passed"] = False
        with self.assertRaises(ValueError):
            serving_row(settings(), report, "a" * 40)

    def test_exact_public_audit_run_metadata(self):
        row = synthetic_row()
        metadata = {"id": 1, "head_sha": "a" * 40, "status": "completed", "conclusion": "success", "event": "workflow_dispatch",
                    "path": ".github/workflows/evaluator-owner-keyword_extraction.yml",
                    "repository": {"full_name": "zkasuran/telegraph-catalog-miners", "private": False}}
        check_run(metadata, row)
        with self.assertRaises(ValueError):
            check_run({**metadata, "head_sha": "b" * 40}, row)
        with self.assertRaises(ValueError):
            check_run({**metadata, "status": "in_progress"}, row)

    def test_exact_restricted_public_lease_and_expiry_controls(self):
        # Public wire-shaped dummy bytes, not a generated SSH key or signing identity.
        key = "ssh-ed25519 " + base64.b64encode(struct.pack(">I", 11) + b"ssh-ed25519" + struct.pack(">I", 32) + b"\0" * 32).decode()
        row = synthetic_row()
        created = datetime(2026, 11, 20, tzinfo=timezone.utc)
        lease = lease_common.lease_for(row, key, "synthetic-test", "1", "a" * 40, created)
        self.assertEqual(lease["candidate_proof_sha256"], lease["runtime_proof_sha256"])
        self.assertEqual(lease["runtime_proof_sha256"], lease["source_proof_sha256"])
        self.assertIn('permitlisten="127.0.0.1:48602"', lease["authorized_keys_line"])
        self.assertIn('permitopen="registration-denied.invalid:1",command="/bin/false"', lease["authorized_keys_line"])
        lease_common.validate_lease(lease, row, created)
        with self.assertRaises(ValueError):
            lease_common.validate_lease(lease, row, lease_common.utc(lease["expires_at"]))
        forged = copy.deepcopy(lease)
        forged["source_proof_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            lease_common.validate_lease(forged, row, created)
        self.assertFalse(hasattr(lease_common, "ANCHORS"))

    def test_remote_guard_runs_before_any_serving_asset_or_key_operation(self):
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError), mock.patch("serve.native.fetch") as fetch:
            serve.prepare(Path("unused-local-root"), settings())
        fetch.assert_not_called()

    def test_literal_prefix_token_and_package_cap_controls(self):
        row = synthetic_row()
        value = {"batch": [{"tokens": [50281, 1, 50282]}], **identity(row)}
        value["input_root"] = input_root(value["batch"])
        self.assertEqual(validate_request(value, row)["input_root"], value["input_root"])
        uppercase = {**value, "adapter_hash": value["adapter_hash"].upper()}
        validate_request(uppercase, row)
        for field in (*identity(row), "input_root"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_request({**value, field: "0x" + value[field]}, row)
        for tokens in ([-1], [True], [50368], [1] * 385, []):
            with self.subTest(tokens=tokens[:3]), self.assertRaises(ValueError):
                validate_request({**identity(row), "batch": [{"tokens": tokens}]}, row)

    def test_serving_supervisor_does_not_raise_the_training_timeout(self):
        self.assertEqual(timeout_for("train", [], {"timeout_seconds": 6900}), 6900)
        self.assertEqual(timeout_for("serve", ["python3", str(Path(serve.__file__))], {}), 19800)
        with self.assertRaises(ValueError):
            timeout_for("serve", ["python3", "arbitrary-model-script.py"], {})

    def test_real_http_primary_path_and_nonfinite_backend_refusal(self):
        class FakeBackend:
            calls = 0
            bits = 1065353216
            def score(self, request):
                self.calls += 1
                return {**identity(row), "input_root": request["input_root"], "results": [{"score": self.bits} for _ in request["batch"]]}
        row = synthetic_row()
        backend = FakeBackend()
        server = BoundedServer(("127.0.0.1", 0), row, backend)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def request(value):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            try:
                connection.request("POST", "/runnable", json.dumps(value), {"Content-Type": "application/json"})
                response = connection.getresponse()
                return response.status, json.loads(response.read(8192))
            finally:
                connection.close()
        try:
            value = {**identity(row), "batch": [{"tokens": [50281, 1, 50282]}]}
            status, reply = request(value)
            self.assertEqual(status, 200)
            self.assertEqual(reply["results"], [{"score": 1065353216}])
            self.assertEqual(request({**value, "adapter_hash": "0x" + value["adapter_hash"]})[0], 400)
            self.assertEqual(backend.calls, 1)
            backend.bits = 2139095040
            self.assertEqual(request(value)[0], 502)
            self.assertTrue(server.fatal.is_set())
            self.assertEqual(request(value)[0], 503)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
