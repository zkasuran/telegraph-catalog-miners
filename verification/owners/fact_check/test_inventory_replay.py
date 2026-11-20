# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Actual source replay and failure controls. No fixtures are training material."""
import copy
import json
import unittest

from data import HERE
from inventory_replay import decode_archive, replay, verify_sources


class InventoryReplayTests(unittest.TestCase):
    def inputs(self):
        manifest = json.loads((HERE / "inventory-manifest.json").read_text())
        raw = (HERE / "inventory-snapshot.tar.gz").read_bytes()
        return raw, manifest, decode_archive(raw, manifest)

    def test_real_full_primary_inventory_replays_without_a_network(self):
        report = replay()
        self.assertEqual(report["population"], 83761)
        self.assertEqual(report["page_count"], 9)
        self.assertEqual(report["network_requests"], 0)
        self.assertFalse(report["source_eligible_claims_verified"])

    def test_missing_tail_and_false_complete_history_are_refused(self):
        _, manifest, files = self.inputs()
        incomplete = dict(files)
        incomplete.pop("page-08.json")
        with self.assertRaisesRegex(ValueError, "missing required"):
            verify_sources(incomplete, manifest)
        reset = dict(files)
        state = json.loads(reset["completion.json"])
        state["original_attempts"] = 0
        reset["completion.json"] = json.dumps(state).encode()
        with self.assertRaisesRegex(ValueError, "resets history"):
            verify_sources(reset, manifest)

    def test_tampered_archive_manifest_and_primary_page_are_refused(self):
        raw, manifest, files = self.inputs()
        with self.assertRaises(ValueError):
            decode_archive(raw[:-1] + bytes([raw[-1] ^ 1]), manifest)
        wrong = copy.deepcopy(manifest)
        wrong["population"] -= 1
        with self.assertRaises(ValueError):
            verify_sources(files, wrong)
        corrupt = dict(files)
        corrupt["page-00.json"] = b"invalid primary response"
        with self.assertRaisesRegex(ValueError, "record bytes"):
            verify_sources(corrupt, manifest)


if __name__ == "__main__":
    unittest.main()
