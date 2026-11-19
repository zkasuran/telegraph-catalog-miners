# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Literal node identity and FP32 score-bit controls. Never normalizes hashes."""
import json
import hashlib
import math
import re
import struct

HASH = re.compile(r"[0-9a-f]{64}")
FOUNDATION = "0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6"
ENGINE = "77643d6e1176ab8744801cef93ccb844d0136ed32b8daa865d541f37d13292de"
TOKENIZER = "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456"
MAX_BODY = 1 << 20
MAX_BATCH = 100
MAX_TOKENS = 256
VOCAB = 50368


def input_root(batch):
    # Counterparty: runtime.EncodeFloatInput, runtime.InputHash and
    # remote.BatchInputRoot. Key order and separators are literal wire bytes.
    leaves = []
    for i, tokens in enumerate(batch):
        raw = json.dumps({"input_ids": tokens, "attention_mask": [1] * len(tokens)}, separators=(",", ":")).encode()
        leaves.append(str(i) + ":" + hashlib.sha256(raw).hexdigest())
    return hashlib.sha256("\n".join(sorted(leaves)).encode()).hexdigest() if leaves else ""


def strict_json(raw):
    if not isinstance(raw, bytes) or len(raw) > MAX_BODY:
        raise ValueError("wire bytes required within ceiling")
    def unique(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("duplicate wire JSON key")
            result[k] = v
        return result
    def invalid(value):
        raise ValueError("nonfinite wire constant: " + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def score_value(bits):
    if type(bits) is not int or not 0 <= bits <= 0xffffffff:
        raise ValueError("uint32 score bits required")
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    if not math.isfinite(value):
        raise ValueError("nonfinite FP32 score")
    return value


def request(batch, package_keccak, input_root):
    if not isinstance(package_keccak, str) or not isinstance(input_root, str) or not HASH.fullmatch(package_keccak) or not HASH.fullmatch(input_root):
        raise ValueError("literal bare lowercase hash required")
    if not isinstance(batch, list) or not 1 <= len(batch) <= MAX_BATCH:
        raise ValueError("batch count ceiling")
    for tokens in batch:
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= MAX_TOKENS:
            raise ValueError("token count ceiling")
        if any(type(t) is not int or not 0 <= t < VOCAB for t in tokens):
            raise ValueError("token type or vocabulary ceiling")
    return {"batch": [{"tokens": t} for t in batch], "foundation_hash": FOUNDATION,
            "tokenizer_hash": TOKENIZER, "adapter_hash": package_keccak, "input_root": input_root}


def response(raw, sent):
    value = strict_json(raw)
    if not isinstance(value, dict):
        raise ValueError("wire response object required")
    for key in ("foundation_hash", "tokenizer_hash", "adapter_hash", "input_root"):
        if value.get(key) != sent[key]:
            raise ValueError("literal reference identity mismatch: " + key)
    results = value.get("results")
    if not isinstance(results, list) or len(results) != len(sent["batch"]):
        raise ValueError("one result per actual request item required")
    bits = []
    for row in results:
        if not isinstance(row, dict) or set(row) != {"score"}:
            raise ValueError("runtime error or unexpected score item")
        score_value(row["score"])
        bits.append(row["score"])
    return bits
