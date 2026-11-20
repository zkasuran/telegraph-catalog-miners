# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Mechanical resource tamper controls with no model/key/SSH execution."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import serving


class ServingAssetTests(unittest.TestCase):
    def test_actual_public_pin_contract_primary_and_changed_hash_refusal(self):
        serving.validate_pins(serving.lease_common.HOST_KEY, dict(serving.contracts.FOUNDATION))
        for name in serving.contracts.FOUNDATION:
            hashes = dict(serving.contracts.FOUNDATION)
            hashes[name] = "0" * 64
            with self.assertRaises(ValueError):
                serving.validate_pins(serving.lease_common.HOST_KEY, hashes)

    def test_changed_foundation_bytes_refused_before_executor_spawn(self):
        with tempfile.TemporaryDirectory(prefix="serving-byte-test-", dir=Path(__file__).resolve().parent) as scratch:
            root = Path(scratch)
            (root / "known_hosts").write_text(serving.lease_common.HOST_KEY)
            (root / "foundation").mkdir()
            for name in serving.contracts.FOUNDATION:
                (root / "foundation" / name).write_bytes(b"mechanical bytes, not model weights")
            with patch("serving.subprocess.Popen") as spawn, patch("serving.Server"), patch("contracts.remote_only"):
                with self.assertRaises(ValueError):
                    serving.engine(root, {"keccak": "0" * 64})
                spawn.assert_not_called()

    def test_changed_host_pin_is_refused_before_executor_spawn(self):
        with tempfile.TemporaryDirectory(prefix="serving-assets-", dir=Path(__file__).resolve().parent) as scratch:
            root = Path(scratch)
            (root / "known_hosts").write_text("mechanical changed public host pin")
            row = {"keccak": "0" * 64}
            # The real engine-start function is exercised. External process and
            # backend server are fake so no model or forwarding operation runs.
            with patch("serving.subprocess.Popen"), patch("serving.Server"), patch("contracts.remote_only"):
                with self.assertRaises(ValueError):
                    serving.engine(root, row)


if __name__ == "__main__":
    unittest.main()
