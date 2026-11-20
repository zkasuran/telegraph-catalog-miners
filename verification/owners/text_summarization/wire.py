# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Literal node batch and finite FP32 contracts, never normalized stand-ins."""
import json
import math
import re
import struct

from data import digest


def json_bytes(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


def input_root(batch):
    leaves = [f"{i}:" + digest(json_bytes({"input_ids": item["tokens"], "attention_mask": [1] * len(item["tokens"])})) for i, item in enumerate(batch)]
    return digest("\n".join(sorted(leaves)).encode()) if leaves else ""


def decode_bits(bits):
    if type(bits) is not int or not 0 <= bits < 1 << 32:
        raise ValueError("invalid FP32 score bits")
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    if not math.isfinite(value):
        raise ValueError("nonfinite runtime score")
    return value


def validate_response(value, request, refusals=False):
    identity = {key: request[key] for key in ("foundation_hash", "adapter_hash", "tokenizer_hash")}
    if any(not isinstance(v, str) or not re.fullmatch(r"[a-f0-9]{64}", v) for v in identity.values()):
        raise ValueError("literal bare lowercase identity required")
    if not isinstance(value, dict) or set(value) != {"results", "input_root", *identity}:
        raise ValueError("literal runtime response fields mismatch")
    if any(value[k] != v for k, v in identity.items()) or value["input_root"] != input_root(request["batch"]):
        raise ValueError("literal runtime identity or input root mismatch")
    results = value["results"]
    if not isinstance(results, list) or len(results) != len(request["batch"]):
        raise ValueError("runtime result count mismatch")
    scores = []
    for item in results:
        if refusals:
            if not isinstance(item, dict) or set(item) != {"error"} or not isinstance(item["error"], str) or not 0 < len(item["error"]) <= 4096:
                raise ValueError("real executor did not refuse invalid tokens")
        else:
            if not isinstance(item, dict) or set(item) != {"score"}:
                raise ValueError("runtime score field mismatch")
            decode_bits(item["score"])
            scores.append(item["score"])
    return scores
