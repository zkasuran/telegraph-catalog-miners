# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Literal wire and registration-report contracts, with no neural imports."""
import math
import struct

FORMATS = ("prose", "json", "report")


def finite_bits(bits):
    if type(bits) is not int or not 0 <= bits <= 0xffffffff:
        raise ValueError("invalid FP32 bits")
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    if not math.isfinite(value):
        raise ValueError("nonfinite FP32 bits")
    return value


def response_bits(body, request):
    for field in ("foundation_hash", "tokenizer_hash", "adapter_hash", "input_root"):
        if body.get(field) != request[field]:
            raise ValueError("literal " + field + " identity mismatch")
    results = body.get("results")
    if not isinstance(results, list) or len(results) != len(request["batch"]):
        raise ValueError("result count mismatch")
    bits = [r["score"] for r in results]
    for b in bits:
        finite_bits(b)
    return bits
