# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Retained literal reply contract. Actual engine audit lives in runtime.py."""
import math
import struct

from assets import remote_only


def validate_reply(obj, identity, count):
    if any(obj.get(k) != v for k, v in identity.items()):
        raise ValueError("literal runtime identity mismatch")
    results = obj.get("results")
    if not isinstance(results, list) or len(results) != count:
        raise ValueError("runtime result count mismatch")
    bits = []
    for r in results:
        if not isinstance(r, dict) or set(r) != {"score"}:
            raise ValueError("runtime refused or malformed result")
        value = r["score"]
        if type(value) is not int or not 0 <= value <= 0xffffffff:
            raise ValueError("runtime score is not literal uint32 FP32 bits")
        if not math.isfinite(struct.unpack("<f", struct.pack("<I", value))[0]):
            raise ValueError("nonfinite runtime score")
        bits.append(value)
    return bits


if __name__ == "__main__":
    remote_only()
    raise SystemExit("Use runtime.py for the complete reference-engine audit")
