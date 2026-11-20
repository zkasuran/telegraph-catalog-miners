# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only identity recovery and lease lifecycle failure controls."""
import copy
import json
import os
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import endpoint_recovery as recovery
import lease_guard as guard
from contracts import package_parts


class EndpointRecoveryTests(unittest.TestCase):
    def test_exact_passing_tensor_payload_descriptors_and_grants_stay_unchanged(self):
        result = recovery.build()
        self.assertTrue(result["tensor_payload_identical"])
        self.assertTrue(result["original_fields_grants_and_descriptors_identical"])
        self.assertNotEqual(result["origin_package_sha256"], result["package_sha256"])
        self.assertEqual(recovery.verify_identity(recovery.ORIGIN.read_bytes(), recovery.RECOVERED.read_bytes()), result["tensor_payload_sha256"])

    def test_changed_tensor_descriptor_model_field_or_upstream_grant_is_refused(self):
        original = recovery.ORIGIN.read_bytes()
        recovered = recovery.RECOVERED.read_bytes()
        with self.assertRaises(ValueError):
            recovery.verify_identity(original, recovered[:-1] + bytes([recovered[-1] ^ 1]))
        header, payload = package_parts(recovered)
        for key, value in (("class", "hybrid"), ("x-licence", "removed")):
            mutated = copy.deepcopy(header)
            mutated["__metadata__"][key] = value
            encoded = json.dumps(mutated).encode()
            with self.assertRaises(ValueError):
                recovery.verify_identity(original, struct.pack("<Q", len(encoded)) + encoded + payload)

    def test_local_recovery_verification_refuses_before_any_neural_work(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            recovery.verify_remote(Path("source-only-fixture"))


class LeaseGuardTests(unittest.TestCase):
    def state(self):
        return {"guard_deadline": 86400, "generation": 0, "lease_expires_epoch": 18000,
                "node_status": "pending", "pending_action": None, "owner_attempt": "source-only"}

    def test_renewal_is_requested_before_expiry_not_after_remote_failure(self):
        self.assertEqual(guard.due(self.state(), 17099), "observe")
        self.assertEqual(guard.due(self.state(), 17100), "dispatch_continuation")

    def test_pending_action_is_resolved_without_another_dispatch(self):
        state = self.state()
        state["pending_action"] = {"key": "a"*64}
        self.assertEqual(guard.due(state, 17999), "resolve_action")
        state["pending_action"] = None
        state["pending_run_id"] = 1
        self.assertEqual(guard.due(state, 17999), "inspect_public_lease")

    def test_guard_stops_at_bounds_and_never_renews_rejected_candidate(self):
        state = self.state()
        self.assertEqual(guard.due(state, 86400), "bounded_stop")
        state["generation"] = 6
        self.assertEqual(guard.due(state, 100), "bounded_stop")
        state["generation"] = 0
        state["node_status"] = "rejected"
        self.assertEqual(guard.due(state, 100), "terminal_stop")

    def test_action_identity_is_stable_and_generation_specific(self):
        first = {"commit": "a"*40, "inputs": {"lease_epoch": "continuation-1"}}
        second = {"commit": "a"*40, "inputs": {"lease_epoch": "continuation-2"}}
        self.assertEqual(guard.action_key("dispatch", first), guard.action_key("dispatch", copy.deepcopy(first)))
        self.assertNotEqual(guard.action_key("dispatch", first), guard.action_key("dispatch", second))

    def test_guard_cannot_send_registration_or_publish(self):
        for action in ("register", "publish", "sign", "reserve_endpoint"):
            with self.assertRaises(ValueError):
                guard.submit(None, None, action, {})

    def test_changed_owner_attempt_stops_before_any_adapter_call(self):
        args = SimpleNamespace(attempt=Path("source-only"))
        with patch.object(guard, "read", return_value={"family": "SENTIMENT_ANALYSIS", "run_id": "other"}), \
                patch.object(guard, "command") as command:
            self.assertFalse(guard.advance(args, {}, self.state()))
            command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
