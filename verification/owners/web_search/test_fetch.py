# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Harmless source-download fixtures. No model assets, network or inference."""
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import remote

PAYLOAD = b"harmless-download-control"
HASH = hashlib.sha256(PAYLOAD).hexdigest()
URL = "https://raw.githubusercontent.com/zkasuran/telegraph-catalog-miners/" + "a" * 40 + "/fixture.bin"


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fetch-control-", dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)
        self.path = self.root / "download.bin"
        self.network = patch.object(remote.urllib.request, "urlopen", side_effect=lambda *args, **kwargs: io.BytesIO(PAYLOAD))
        self.mock = self.network.start()

    def tearDown(self):
        self.network.stop()
        self.temp.cleanup()

    def test_primary_download_and_verified_cache_smoke(self):
        remote.fetch(URL, self.path, HASH, 100)
        self.assertEqual(self.path.read_bytes(), PAYLOAD)
        remote.fetch(URL, self.path, HASH, 100)
        self.assertEqual(self.mock.call_count, 1)

    def test_cached_symlink_refused_before_read_or_network(self):
        victim = self.root / "cache-owner.bin"
        victim.write_bytes(PAYLOAD)
        self.path.symlink_to(victim)
        with self.assertRaises(ValueError):
            remote.fetch(URL, self.path, HASH, 100)
        self.mock.assert_not_called()
        self.assertEqual(victim.read_bytes(), PAYLOAD)

    def test_partial_symlink_never_followed_or_removed(self):
        victim = self.root / "owned-victim.bin"
        victim.write_bytes(b"original-victim")
        partial = self.path.with_name(self.path.name + ".partial")
        partial.symlink_to(victim)
        remote.fetch(URL, self.path, HASH, 100)
        self.assertFalse(self.path.is_symlink())
        self.assertEqual(self.path.read_bytes(), PAYLOAD)
        self.assertEqual(victim.read_bytes(), b"original-victim")
        self.assertTrue(partial.is_symlink())

    def test_preexisting_partial_checkpoint_preserved(self):
        partial = self.path.with_name(self.path.name + ".partial")
        partial.write_bytes(b"prior-failed-download")
        remote.fetch(URL, self.path, HASH, 100)
        self.assertEqual(partial.read_bytes(), b"prior-failed-download")
        self.assertEqual(self.path.read_bytes(), PAYLOAD)

    def test_parent_symlink_refused_before_network(self):
        actual = self.root / "actual"
        actual.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(actual, target_is_directory=True)
        with self.assertRaises(ValueError):
            remote.fetch(URL, alias / "download.bin", HASH, 100)
        self.mock.assert_not_called()

    def test_hash_failure_preserves_prior_destination(self):
        self.path.write_bytes(b"old-destination")
        with self.assertRaises(ValueError):
            remote.fetch(URL, self.path, "0" * 64, 100)
        self.assertEqual(self.path.read_bytes(), b"old-destination")
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["download.bin"])

    def test_size_failure_cleans_only_owned_staging(self):
        self.path.write_bytes(b"old-destination")
        with self.assertRaises(ValueError):
            remote.fetch(URL, self.path, HASH, 2)
        self.assertEqual(self.path.read_bytes(), b"old-destination")
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["download.bin"])

    def test_fifo_and_bad_hash_identity_refused(self):
        import os
        os.mkfifo(self.path)
        with self.assertRaises(ValueError):
            remote.fetch(URL, self.path, HASH, 100)
        self.mock.assert_not_called()
        self.path.unlink()
        for value in ("A" * 64, "x" * 64, "0x" + HASH):
            with self.assertRaises(ValueError):
                remote.fetch(URL, self.path, value, 100)
        self.mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
