# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Artificial container controls only, never candidate or neural evaluation data."""
import copy
import hashlib
import json
import struct
import unittest

from release_gate import checked_bytes
from test_ready import artificial_report


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.terms = {"x-licence": "artificial-test-only", "x-upstream": "fixture grant"}
        header = json.dumps({"__metadata__": self.terms}).encode()
        self.raw = struct.pack("<Q", len(header)) + header + b"artificial payload"
        self.report = artificial_report()
        self.report.update(package_bytes=len(self.raw), package_sha256=hashlib.sha256(self.raw).hexdigest())

    def test_bound_artificial_container(self):
        self.assertFalse(checked_bytes(self.raw, self.report, self.terms)["neural_loaded"])

    def test_changed_package_and_byte_count_refused(self):
        with self.assertRaises(ValueError):
            checked_bytes(self.raw[:-1] + b"x", self.report, self.terms)
        with self.assertRaises(ValueError):
            checked_bytes(self.raw, {**self.report, "package_bytes": len(self.raw) + 1}, self.terms)

    def test_missing_grant_refused(self):
        with self.assertRaises(ValueError):
            checked_bytes(self.raw, self.report, {**self.terms, "other": "retained condition"})

    def test_failed_reference_report_refused(self):
        report = copy.deepcopy(self.report)
        report["runtime"]["reference_engine_passed"] = False
        with self.assertRaises(ValueError):
            checked_bytes(self.raw, report, self.terms)

    def test_invalid_header_refused(self):
        raw = struct.pack("<Q", 1 << 21) + self.raw[8:]
        report = {**self.report, "package_sha256": hashlib.sha256(raw).hexdigest()}
        with self.assertRaises(ValueError):
            checked_bytes(raw, report, self.terms)


if __name__ == "__main__":
    unittest.main()
