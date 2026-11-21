# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Actual audit wrapper with synthetic auditor output. No package/model imports."""
import copy
import io
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

import audit_dependencies as audit


class DependencyAuditTests(unittest.TestCase):
    def setUp(self):
        self.lock = (Path(__file__).parent / "requirements.lock").read_text()
        audited, self.mapped = audit.requirements(self.lock)
        self.rows = [{"name": name, "version": version, "vulns": []}
                     for name, version in re.findall(r"^([^=]+)==([^\n]+)", audited, re.M)]

    def invoke(self, rows=None, returncode=0, raw=None, lock=None):
        result = SimpleNamespace(stdout=raw if raw is not None else json.dumps({"dependencies": self.rows if rows is None else rows, "fixes": []}),
                                 stderr="synthetic auditor", returncode=returncode)
        with tempfile.TemporaryDirectory(prefix="dependency-source-contract-") as tmp:
            root = Path(tmp)
            (root / "lock").write_text(self.lock if lock is None else lock)
            argv = ["audit_dependencies.py", "--lock", str(root / "lock"), "--out", str(root / "audit.in"), "--report", str(root / "report.json")]
            with patch.object(audit.sys, "argv", argv), patch.object(audit.subprocess, "run", return_value=result) as process, redirect_stdout(io.StringIO()):
                code = audit.main()
            self.assertEqual(process.call_count, 1)
            return code, json.loads((root / "report.json").read_text())

    def test_complete_63_release_primary_path(self):
        self.assertEqual(len(self.rows), 63)
        code, report = self.invoke()
        self.assertEqual((code, report["verdict"]), (0, "PASS"))
        self.assertEqual(report.get("coverage_verified"), True)
        self.assertEqual(report.get("expected_dependencies"), 63)
        self.assertEqual(report.get("reported_dependencies"), 63)

    def test_empty_and_partial_audits_fail(self):
        for rows in ([], self.rows[:-1]):
            with self.subTest(rows=len(rows)):
                code, report = self.invoke(rows)
                self.assertEqual((code, report["verdict"]), (1, "FAIL"))

    def test_duplicates_and_unexpected_releases_fail(self):
        for rows in (self.rows + [self.rows[0]], self.rows + [{"name": "unexpected", "version": "1.0", "vulns": []}]):
            with self.subTest(rows=len(rows)):
                self.assertEqual(self.invoke(rows)[0], 1)

    def test_substituted_version_fails(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["version"] = "0.0.0"
        self.assertEqual(self.invoke(rows)[0], 1)

    def test_missing_and_malformed_vulnerability_data_fail(self):
        for vulns in (None, False, [{}], ""):
            rows = copy.deepcopy(self.rows)
            rows[0]["vulns"] = vulns
            with self.subTest(vulns=vulns):
                self.assertEqual(self.invoke(rows)[0], 1)
        rows = copy.deepcopy(self.rows)
        del rows[0]["vulns"]
        self.assertEqual(self.invoke(rows)[0], 1)

    def test_skips_vulnerabilities_and_nonzero_exit_remain_failures(self):
        skipped = copy.deepcopy(self.rows)
        skipped[0] = {"name": skipped[0]["name"], "skip_reason": "synthetic skip"}
        self.assertEqual(self.invoke(skipped)[0], 1)
        vulnerable = copy.deepcopy(self.rows)
        vulnerable[0]["vulns"] = [{"id": "SYNTHETIC-1", "fix_versions": []}]
        self.assertEqual(self.invoke(vulnerable)[0], 1)
        self.assertEqual(self.invoke(returncode=2)[0], 1)

    def test_canonical_names_and_alias_duplicates(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["name"] = rows[0]["name"].upper()
        self.assertEqual(self.invoke(rows)[0], 0)
        with self.assertRaises(ValueError):
            audit.requirements("Friendly_Bard==1.0\nfriendly.bard==1.0\n")

    def test_cpu_variant_mapping_is_retained_not_generalized(self):
        text, mapping = audit.requirements("torch==2.15.0+cpu\n")
        self.assertEqual(text, "torch==2.15.0\n")
        self.assertEqual(mapping["torch"]["installed_variant"], "2.15.0+cpu")
        with self.assertRaises(ValueError):
            audit.requirements("torch==2.15.0+other\n")

    def test_unsupported_or_unpinned_lock_rows_refused(self):
        for content in ("alpha==1.0\n-r other.in\n", "alpha==1.0\nhttps://example.invalid/package.whl\n", "alpha>=1.0\n"):
            with self.subTest(content=content), self.assertRaises(ValueError):
                audit.requirements(content)

    def test_duplicate_or_malformed_json_fails(self):
        valid = json.dumps({"dependencies": self.rows, "fixes": []})
        duplicate = valid[:-1] + ', "dependencies": []}'
        for raw in (duplicate, "[]", '{"dependencies": null, "fixes": []}'):
            with self.subTest(raw=raw[:40]):
                self.assertEqual(self.invoke(raw=raw)[0], 1)


if __name__ == "__main__":
    unittest.main()
