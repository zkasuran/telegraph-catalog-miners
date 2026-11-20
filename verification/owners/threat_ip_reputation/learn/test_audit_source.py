# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Mechanical source-boundary tests. Fixtures are not attack observations."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from audit_source import eligible, baseline_files, FORMATS
import contracts


def row(ip, asn, last="2026-11-20T01:00:00Z"):
    return {"ip": ip, "asn": asn, "last_seen": last}


class AuditSourceTests(unittest.TestCase):
    def test_original_source_archive_bytes_and_failure_controls(self):
        baseline = Path(__file__).resolve().parent / "audit-base"
        manifest = json.loads((baseline / "source-manifest.json").read_text())
        with baseline_files(baseline) as extracted:
            raw = {name: (extracted / name).read_bytes() for name in FORMATS}
        for name in FORMATS:
            self.assertEqual(contracts.sha(raw[name]), manifest["files"][name]["sha256"])
        with tempfile.TemporaryDirectory(prefix="audit-archive-test-", dir=baseline.parent) as scratch:
            root = Path(scratch)
            (root / "source-manifest.json").write_text(json.dumps(manifest))
            with zipfile.ZipFile(root / "original-source.zip", "w") as out:
                for name in FORMATS:
                    out.writestr(name, raw[name])
                out.writestr("../outside", b"mechanical refusal control")
            with self.assertRaises(ValueError):
                with baseline_files(root):
                    self.fail("unsafe archive accepted")
            with zipfile.ZipFile(root / "original-source.zip", "w") as out:
                for name in FORMATS:
                    out.writestr(name, raw[name] + b"tampered" if name == "blocklist.json" else raw[name])
            with self.assertRaises(ValueError):
                with baseline_files(root):
                    self.fail("changed original source bytes accepted")

    def test_real_audit_entry_refuses_local_neural_work(self):
        path = Path(__file__).resolve().parent / "remote_audit.py"
        process = subprocess.run([sys.executable, "-W", "error", str(path), "--root", "unused", "--candidate", "unused"],
                                 capture_output=True, text=True, timeout=10, env={**os.environ, "GITHUB_ACTIONS": "false"})
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("remote", process.stderr.lower())
        self.assertNotIn("No module named", process.stderr)

    def test_primary_fresh_disjoint_source(self):
        groups, _, _ = eligible([row("8.8.8.8", "AS1")], [row("9.9.9.9", "AS2")], {"8.8.8.8"}, "2026-11-20T00:41:05+00:00")
        self.assertEqual(len(groups), 1)

    def test_old_observation_or_equal_freeze_is_not_fresh(self):
        for at in ("2026-11-20T00:00:00Z", "2026-11-20T00:41:05Z"):
            groups, excluded, _ = eligible([], [row("9.9.9.9", "AS2", at)], set(), "2026-11-20T00:41:05+00:00")
            self.assertEqual(groups, [])
            self.assertEqual(excluded["no_post_freeze_observation"], 1)

    def test_global_transitive_old_new_asn_bridge_is_refused(self):
        old = [row("8.8.8.8", "AS1"), row("9.9.9.9", "AS2")]
        new = [row("9.9.9.9", "AS1"), row("11.0.0.1", "AS2")]
        groups, excluded, _ = eligible(old, new, {"8.8.8.8"}, "2026-11-20T00:41:05+00:00")
        self.assertEqual(groups, [])
        self.assertEqual(excluded["development_overlap"], 1)

    def test_network_and_unknown_asn_refusals(self):
        groups, excluded, _ = eligible([row("8.8.8.8", "AS1")], [row("8.8.8.9", "AS2"), row("9.9.9.9", "AS0")], {"8.8.8.8"}, "2026-11-20T00:41:05+00:00")
        self.assertEqual(groups, [])
        self.assertEqual(excluded["development_overlap"], 1)
        self.assertEqual(excluded["unknown_asn"], 1)


if __name__ == "__main__":
    unittest.main()
