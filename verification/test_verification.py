# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import datetime as dt
import hashlib
import io
import json
import tempfile
import struct
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

    def test_continuation_only_accepts_its_exact_existing_registration(self):
        old={**self.entry,"registered_at":"2026-11-17T00:05:00Z"}
        row={**self.row,"existing_registration_id":100}
        self.assertEqual(serve.find_decision([old],row,self.start)["registration_id"],100)
        self.assertIsNone(serve.find_decision([{**old,"registration_id":99}],row,self.start))

    def test_manifest_has_only_public_exact_bindings(self):
        rows = json.loads((Path(__file__).parent / "champions.json").read_text())["rows"]
        self.assertEqual(len(rows), 13)
        self.assertEqual(len({r["family"] for r in rows}), 13)
        for row in rows:
            self.assertEqual(set(row), {"family", "slot", "file", "url", "keccak", "sha256", "class","origin_url","origin_keccak","origin_sha256","tensor_payload_sha256","forward_port","runnable_url"})
            self.assertEqual(len(row["keccak"]), 64)
            self.assertEqual(len(row["sha256"]), 64)
            self.assertNotIn("/home/", row["url"])

class TensorCopy(unittest.TestCase):
    def test_metadata_extensions_preserve_payload_and_model_fields(self):
        header={"__metadata__":{"abi":"lora-v1","rank":"8"},"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}
        def package(h,payload):
            text=json.dumps(h).encode()
            return struct.pack("<Q",len(text))+text+payload
        payload=struct.pack("<f",1.25)
        old=package(header,payload)
        new=json.loads(json.dumps(header));new["__metadata__"]["x-recovery"]="test"
        self.assertEqual(prepare.verify_tensor_copy(old,package(new,payload)),hashlib.sha256(payload).hexdigest())
        with self.assertRaises(RuntimeError):
            prepare.verify_tensor_copy(old,package(new,struct.pack("<f",1.5)))
        new["__metadata__"]["rank"]="16"
        with self.assertRaises(RuntimeError):
            prepare.verify_tensor_copy(old,package(new,payload))

if __name__ == "__main__":
    unittest.main()
