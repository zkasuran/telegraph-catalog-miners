# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import datetime as dt
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import prepare
import serve

class Downloads(unittest.TestCase):
    def test_correct_bytes_pass_and_tampered_bytes_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model"
            expected = hashlib.sha256(b"verified bytes").hexdigest()
            with patch.object(prepare.urllib.request, "urlopen", return_value=io.BytesIO(b"verified bytes")):
                prepare.fetch("https://example.test/model", path, expected)
            with patch.object(prepare.urllib.request, "urlopen", return_value=io.BytesIO(b"tampered bytes")):
                with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                    prepare.fetch("https://example.test/model", path, expected)

    def test_size_limit_rejects_before_full_download(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(prepare.urllib.request, "urlopen", return_value=io.BytesIO(b"oversized")):
            with self.assertRaisesRegex(RuntimeError, "ceiling"):
                prepare.fetch("https://example.test/model", Path(folder) / "model", ceiling=3)

class Decisions(unittest.TestCase):
    def setUp(self):
        self.start = dt.datetime(2026, 11, 18, tzinfo=dt.timezone.utc)
        self.row = {"family": "CODE_REVIEW", "keccak": "a" * 64}
        self.entry = {"author_address": serve.OWNER, "wasm_hash": "a" * 64, "registered_at": "2026-11-18T00:05:00Z", "activation_status": "active", "registration_id": 100}

    def test_old_or_other_candidates_do_not_end_the_test(self):
        for change in ({"registered_at": "2026-11-17T00:05:00Z"}, {"author_address": "other"}, {"wasm_hash": "b" * 64}, {"activation_status": "pending"}):
            self.assertIsNone(serve.find_decision([{**self.entry, **change}], self.row, self.start))

    def test_new_exact_candidate_ends_with_real_status(self):
        self.assertEqual(serve.find_decision([self.entry], self.row, self.start)["status"], "active")
        rejected = {**self.entry, "activation_status": "rejected"}
        self.assertEqual(serve.find_decision([rejected], self.row, self.start)["status"], "rejected")

    def test_manifest_has_only_public_exact_bindings(self):
        rows = json.loads((Path(__file__).parent / "champions.json").read_text())["rows"]
        self.assertEqual(len(rows), 13)
        self.assertEqual(len({r["family"] for r in rows}), 13)
        for row in rows:
            self.assertEqual(set(row), {"family", "slot", "file", "url", "keccak", "sha256", "class"})
            self.assertEqual(len(row["keccak"]), 64)
            self.assertEqual(len(row["sha256"]), 64)
            self.assertNotIn("/home/", row["url"])

if __name__ == "__main__":
    unittest.main()
