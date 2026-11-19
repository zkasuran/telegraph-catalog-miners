# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Counterparty string contract. Never strip a hash prefix to manufacture equality."""
import math
import struct


def bits_value(bits):
    if isinstance(bits, bool) or not isinstance(bits, int) or not 0 <= bits < 2**32:
        raise ValueError("invalid FP32 bits")
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    if not math.isfinite(value):
        raise ValueError("nonfinite FP32 score")
    return value


def check_response(reply, expected, count):
    # The primary BatchIdentity.Equal uses EqualFold, not prefix removal.
    for field in ("foundation_hash", "adapter_hash", "tokenizer_hash"):
        value = reply.get(field)
        if not isinstance(value, str) or value.casefold() != expected[field].casefold():
            raise ValueError("literal runtime identity mismatch: " + field)
    if "input_root" in expected:
        value = reply.get("input_root")
        if not isinstance(value, str) or value.casefold() != expected["input_root"].casefold():
            raise ValueError("literal runtime input root mismatch")
    rows = reply.get("results")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("wrong runtime result count")
    results = []
    for row in rows:
        if not isinstance(row, dict) or "error" in row or "score" not in row:
            raise ValueError("runtime item failure")
        bits_value(row["score"])
        results.append(row["score"])
    return results
