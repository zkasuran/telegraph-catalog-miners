# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded opaque checkpoint identity and grant checks, never model loading."""
import json
import struct

from common import sha


def check_checkpoint(raw, expected_sha, parent_source):
    if not 9 <= len(raw) <= 64 << 20 or sha(raw) != expected_sha:
        raise ValueError("checkpoint byte identity mismatch")
    size = struct.unpack("<Q", raw[:8])[0]
    if not 0 < size <= 1 << 20 or size + 8 > len(raw):
        raise ValueError("checkpoint header ceiling")
    head = json.loads(raw[8:8+size])
    meta = head.get("__metadata__", {})
    if meta.get("x-terms-source-manifest-sha256") != parent_source:
        raise ValueError("checkpoint source lineage mismatch")
    for key in ("sand", "apache", "gpt2", "kit", "cc_by_3", "cc_by_4", "source-attribution"):
        if not isinstance(meta.get("x-terms-" + key), str) or not meta["x-terms-" + key]:
            raise ValueError("checkpoint omitted original terms or attribution")
    tensors = {k: v for k, v in head.items() if k != "__metadata__"}
    if not tensors or any(v.get("dtype") != "F32" for v in tensors.values()):
        raise ValueError("checkpoint does not contain expected FP32 learned descriptors")
    return {"sha256": expected_sha, "bytes": len(raw), "descriptors": len(tensors), "source_manifest_sha256": parent_source,
            "tensor_payload_interpreted": False}
