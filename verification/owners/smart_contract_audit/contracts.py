# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Literal wire and registration-report contracts, with no neural imports."""
import math
import struct

FORMATS = ("prose", "json", "report")


def training_pairs(rows):
    """Honor the pinned trainer's optional-margin default without editing source data.

    train_adapter._pair supplies margin=0.0 before PairCollator indexes it.
    Our source inventory is not that loader's output, so adapt it at this boundary.
    Literal prompt/answer strings and all provenance fields remain unchanged.
    """
    result = []
    for row in rows:
        margin = row.get("margin", 0.0)
        if type(margin) not in (int, float) or not math.isfinite(margin):
            raise ValueError("finite numeric preference margin required")
        if any(not isinstance(row.get(field), str) for field in ("prompt", "chosen", "rejected")):
            raise ValueError("literal preference strings required")
        result.append({**row, "margin": float(margin)})
    return result


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


def main():
    """Non-neural smoke for the actual source-to-collator input boundary."""
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True, type=Path)
    path = parser.parse_args().inventory
    raw = path.read_bytes()
    if len(raw) > 2 << 20:
        raise ValueError("source inventory ceiling")
    rows = json.loads(raw)
    selected = [r for r in rows if r["split"] in ("train", "validation")]
    prepared = training_pairs(selected)
    if path.read_bytes() != raw or any(any(p[key] != r[key] for key in r) for p, r in zip(prepared, selected)):
        raise ValueError("collator preparation changed source or literal fields")
    print(json.dumps({"prepared_train_validation_pairs": len(prepared), "defaulted_zero_margins": sum("margin" not in r and p["margin"] == 0.0 for p, r in zip(prepared, selected)), "source_bytes_and_literal_fields_unchanged": True}))


if __name__ == "__main__":
    main()
