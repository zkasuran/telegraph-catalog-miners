# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import json
import struct
import unittest

from preserve import stamp


class PreserveTests(unittest.TestCase):
    def test_checkpoint_terms_do_not_change_tensors_or_existing_metadata(self):
        header = {"__metadata__": {"format": "pt"}, "weight": {
            "dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
        encoded = json.dumps(header).encode()
        payload = struct.pack("<f", 1.25)
        raw = struct.pack("<Q", len(encoded)) + encoded + payload
        out = stamp(raw, {"sand": "terms", "notice": "upstream"})
        size = struct.unpack("<Q", out[:8])[0]
        result = json.loads(out[8:8 + size])
        self.assertEqual(result["weight"], header["weight"])
        self.assertEqual(result["__metadata__"]["format"], "pt")
        self.assertEqual(result["__metadata__"]["x-terms-sand"], "terms")
        self.assertEqual(out[8 + size:], payload)

    def test_invalid_header_cannot_be_stamped(self):
        for value in (b"", struct.pack("<Q", 1 << 22) + b"{}"):
            with self.assertRaises(ValueError):
                stamp(value, {})
