# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Actual source artifact and mutation controls. No network or neural imports."""
import copy
import json
import unittest
from pathlib import Path

from lease_continuation import dispatch_payload, validate

HERE = Path(__file__).resolve().parent


class LeaseTests(unittest.TestCase):
    def fixture(self):
        root = HERE / "continuations/lease_44827170108"
        return json.loads((root / "continuation.json").read_bytes()), (root / "registration-ready.json").read_bytes(), (root / "previous-lease.json").read_bytes()

    def test_actual_continuation_keeps_original_audit_and_runtime(self):
        value, report, lease = self.fixture()
        actual = validate(value, report, lease)
        self.assertEqual(actual["run_id"], "44827170108")
        self.assertEqual(report, (HERE / "candidate/registration-ready.json").read_bytes())
        request = dispatch_payload(value, report, lease, "a" * 40)
        self.assertEqual(request["commit"], value["serving_commit"])
        self.assertEqual(request["inputs"]["mode"], "serve")
        self.assertEqual(request["inputs"]["report_sha256"], value["report_sha256"])
        self.assertTrue(request["inputs"]["report_url"].endswith(value["report_path"]))

    def test_report_and_lease_byte_mutations_refuse(self):
        value, report, lease = self.fixture()
        for raw_report, raw_lease in ((report + b"\n", lease), (report, lease + b"\n")):
            with self.assertRaises(ValueError):
                validate(value, raw_report, raw_lease)

    def test_wrong_package_audit_executor_and_prior_run_refuse(self):
        value, report, lease = self.fixture()
        for field, replacement in (("package_sha256", "f" * 64), ("package_keccak", "f" * 64),
                                   ("audit_run_id", 1), ("audit_commit", "f" * 40), ("serving_commit", "f" * 40),
                                   ("previous_serving_run_id", 1), ("family", "OTHER")):
            changed = copy.deepcopy(value)
            changed[field] = replacement
            with self.assertRaises(ValueError):
                validate(changed, report, lease)

    def test_unzoned_expiry_path_escape_and_missing_grant_refuse(self):
        value, report, lease = self.fixture()
        for field, replacement in (("previous_lease_expires_at", "2026-11-21T03:14:42"), ("report_path", "../registration-ready.json"),
                                   ("license", ""), ("generation", "dummy_counter"), ("package_url", "https://example.invalid/model")):
            with self.assertRaises(ValueError):
                validate({**value, field: replacement}, report, lease)

    def test_unconfirmed_publication_cannot_build_dispatch(self):
        value, report, lease = self.fixture()
        for commit in ("master", "", "0x" + "a" * 40):
            with self.assertRaises(ValueError):
                dispatch_payload(value, report, lease, commit)


if __name__ == "__main__":
    unittest.main()
