# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Mechanical closure controls, never attack observations or audit pairs."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audit_source import eligible


class HistoryTests(unittest.TestCase):
    def test_historical_rows_never_become_current_audit_candidates(self):
        row = {"ip": "9.9.9.9", "asn": "AS2", "last_seen": "2026-11-20T01:00:00Z"}
        selected, excluded, _ = eligible([], [], set(), "2026-11-20T00:41:05+00:00", history_rows=[row])
        self.assertEqual(selected, [])
        self.assertEqual(excluded["no_post_freeze_observation"], 1)

    def test_missing_history_anchor_refuses_incomplete_inventory(self):
        from audit_source import collect_history
        with tempfile.TemporaryDirectory(prefix="history-inventory-test-", dir=Path(__file__).resolve().parent) as scratch:
            with patch("audit_source.get", return_value=json.dumps([]).encode()) as fetch:
                with self.assertRaises(ValueError):
                    collect_history("a" * 40, Path(scratch))
                self.assertEqual(fetch.call_count, 1)

    def test_disappearing_historical_asn_bridge_stays_excluded(self):
        old = [{"ip": "8.8.8.8", "asn": "AS1", "last_seen": "2026-11-19T23:00:00Z"}]
        current = [{"ip": "9.9.9.9", "asn": "AS2", "last_seen": "2026-11-20T01:00:00Z"}]
        history = [{"ip": "8.8.8.8", "asn": "AS2", "last_seen": "2026-11-20T00:00:00Z"}]
        # Exercise the retained original function before its optional history
        # argument exists, so the same semantic assertion proves the defect.
        kwargs = {"history_rows": history} if "history_rows" in inspect.signature(eligible).parameters else {}
        selected, excluded, _ = eligible(old, current, {"8.8.8.8"}, "2026-11-20T00:41:05+00:00", **kwargs)
        self.assertEqual(selected, [], "a past ASN bridge cannot become independent when it disappears")
        self.assertEqual(excluded["development_overlap"], 1)


if __name__ == "__main__":
    unittest.main()
