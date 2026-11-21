# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Literal readback, full-list matching and real restricted-lease controls."""
import copy
import hashlib
import importlib.util
import io
import json
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from lease_readiness import activation_payload, continued_dispatch, lease_bytes, select_artifact, select_run

HERE = Path(__file__).resolve().parent


class ReadinessTests(unittest.TestCase):
    def fixture(self):
        # Public fixture only. No private key or actual new run is manufactured.
        lease = json.loads((HERE / "continuations/lease_44827170108/previous-lease.json").read_bytes())
        candidate = json.loads((HERE / "candidate/registration-ready.json").read_bytes())
        candidate["report_sha256"] = hashlib.sha256((HERE / "candidate/registration-ready.json").read_bytes()).hexdigest()
        row = {"family": lease["family"], "owner": lease["owner"], "slot": lease["slot"], "forward_port": lease["forward_port"],
               "runnable_url": lease["runnable_url"], "keccak": candidate["package_keccak"], "sha256": candidate["package_sha256"],
               "existing_registration_id": 0, **{kind + "_proof": {"sha256": candidate["report_sha256"]} for kind in ("candidate", "runtime", "source")}}
        spec = importlib.util.spec_from_file_location("primary_lease_contract_test", HERE / "support/registration/common.py")
        common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common)
        run = {"id": int(lease["run_id"]), "head_sha": lease["commit"], "status": "in_progress"}
        return lease, candidate, row, common, run

    def zip(self, entries):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, raw in entries:
                archive.writestr(name, raw)
        raw = buffer.getvalue()
        return raw, {"id": 17, "size_in_bytes": len(raw), "digest": "sha256:" + hashlib.sha256(raw).hexdigest()}

    def test_capacity_wait_does_not_create_a_run_or_reset_request(self):
        envelope = {"key": "a" * 64, "submitted_at": 123, "request": {"family": "CONTENT_MODERATION", "action": "reconcile", "payload": {"operation_key": "b" * 64}}}
        receipt = {"key": envelope["key"], "family": "CONTENT_MODERATION", "request_sequence": 123, "status": "waiting_capacity", "result": {"status": "capacity_wait", "retry_allowed": True}}
        self.assertIsNone(continued_dispatch(receipt, envelope, "b" * 64))
        for changed in ({**receipt, "request_sequence": 124}, {**receipt, "family": "OTHER"}, {**receipt, "status": "unknown"}):
            with self.assertRaises(ValueError):
                continued_dispatch(changed, envelope, "b" * 64)

    def test_full_run_listing_identity_and_ambiguity_controls(self):
        payload = {"workflow": ".github/workflows/evaluator-owner-content_moderation.yml", "commit": "a" * 40}
        row = {"id": 17, "head_sha": "a" * 40, "event": "workflow_dispatch", "path": payload["workflow"], "created_at": "2026-11-21T03:00:00Z"}
        start = datetime(2026, 11, 21, 3, tzinfo=timezone.utc).timestamp()
        self.assertEqual(select_run([{**row, "id": 1, "created_at": "2026-11-20T22:00:00Z"}, row], payload, start)["id"], 17)
        for rows in ([row, {**row, "id": 18}], [{**row, "head_sha": "f" * 40}], [{**row, "path": "other.yml"}]):
            with self.assertRaises(ValueError):
                select_run(rows, payload, start)

    def test_actual_one_file_zip_roundtrip_and_whole_hash_refusal(self):
        lease, _, _, _, _ = self.fixture()
        raw, artifact = self.zip([("lease.json", json.dumps(lease).encode())])
        self.assertEqual(lease_bytes(raw, artifact)[0], lease)
        with self.assertRaises(ValueError):
            lease_bytes(raw + b"x", artifact)

    def test_private_extra_traversal_duplicate_json_and_size_refuse(self):
        for entries in ([("lease.json", b"{}"), ("forward-key", b"not a key")], [("../lease.json", b"{}")],
                        [("./lease.json", b"{}")], [("lease.json", b'{"id":1,"id":2}')], [("lease.json", b"x" * 8193)]):
            raw, artifact = self.zip(entries)
            with self.assertRaises(ValueError):
                lease_bytes(raw, artifact)

    def test_artifact_missing_duplicate_wrong_run_and_digest_refuse(self):
        self.assertIsNone(select_artifact([], 17))
        row = {"id": 29, "name": "owner-lease-content_moderation-17", "workflow_run": {"id": 17}, "expired": False, "size_in_bytes": 100, "digest": "sha256:" + "a" * 64}
        self.assertEqual(select_artifact([row], 17)["id"], 29)
        for rows in ([row, copy.deepcopy(row)], [{**row, "workflow_run": {"id": 18}}], [{**row, "digest": "missing"}], [{**row, "expired": True}]):
            with self.assertRaises(ValueError):
                select_artifact(rows, 17)

    def test_primary_restricted_lease_to_exact_activation_and_mutations(self):
        lease, candidate, row, common, run = self.fixture()
        artifact = {"id": 17, "workflow_run": {"id": run["id"]}}
        clock = datetime(2026, 11, 21, 3, tzinfo=timezone.utc)
        payload = activation_payload(lease, run, artifact, candidate, common.validate_lease, row, clock)
        self.assertEqual(payload["serving_run_id"], run["id"])
        for changed in ({**lease, "authorized_keys_line": "unrestricted fixture"}, {**lease, "runtime_proof_sha256": "f" * 64}, {**lease, "run_id": "1"}):
            with self.assertRaises(ValueError):
                activation_payload(changed, run, artifact, candidate, common.validate_lease, row, clock)
        with self.assertRaises(ValueError):
            activation_payload(lease, {**run, "status": "completed"}, artifact, candidate, common.validate_lease, row, clock)
        with self.assertRaises(ValueError):
            activation_payload(lease, run, artifact, candidate, common.validate_lease, row, datetime(2026, 11, 21, 4, tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()
