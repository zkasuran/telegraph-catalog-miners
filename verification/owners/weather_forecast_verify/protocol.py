# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Literal reference-engine batch identity and finite FP32 controls, no neural imports."""
import hashlib
import json
import math
import re
import struct
import urllib.request


def bare_hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("literal bare lowercase hash required, not a normalized stand-in")
    return value


def decode_bits(bits):
    if type(bits) is not int or not 0 <= bits <= 0xffffffff:
        raise ValueError("invalid FP32 score bits")
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    if not math.isfinite(value):
        raise ValueError("nonfinite runtime score")
    return value


def input_root(batch):
    # Literal remote.BatchInputRoot -> runtime.InputHash(EncodeFloatInput) -> InputRoot.
    leaves = []
    for index, tokens in enumerate(batch):
        if any(type(token) is not int for token in tokens):
            raise ValueError("literal integer token required")
        encoded = ('{"input_ids":[' + ','.join(str(token) for token in tokens)
                   + '],"attention_mask":[' + ','.join('1' for _ in tokens) + ']}').encode()
        leaves.append(str(index) + ':' + hashlib.sha256(encoded).hexdigest())
    return hashlib.sha256('\n'.join(sorted(leaves)).encode()).hexdigest() if leaves else ""


def check_reply(value, batch, identities):
    for name in ("foundation_hash", "adapter_hash", "tokenizer_hash"):
        if value.get(name) != bare_hash(identities[name]):
            raise ValueError("literal runtime identity mismatch: " + name)
    if value.get("input_root") != input_root(batch):
        raise ValueError("literal batch input root mismatch")
    results = value.get("results")
    if not isinstance(results, list) or len(results) != len(batch):
        raise ValueError("wire result count mismatch")
    bits = []
    for result in results:
        if not isinstance(result, dict) or set(result) != {"score"}:
            raise ValueError("runtime refused a scored token vector")
        decode_bits(result["score"])
        bits.append(result["score"])
    return bits


def request_value(url, batch, identities):
    payload = {"batch": [{"tokens": tokens} for tokens in batch], **identities, "input_root": input_root(batch)}
    request = urllib.request.Request(url, data=json.dumps(payload, allow_nan=False).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        raw = response.read((1 << 20) + 1)
    if len(raw) > 1 << 20:
        raise ValueError("bounded runtime reply exceeded")
    return json.loads(raw)


def score(url, batch, identities):
    return check_reply(request_value(url, batch, identities), batch, identities)


def negative_identities(url, batch, identities):
    results = []
    for name in ("foundation_hash", "adapter_hash", "tokenizer_hash"):
        wrong = {**identities, name: "0" * 64}
        value = request_value(url, batch, wrong)
        try:
            check_reply(value, batch, wrong)
        except ValueError:
            # Real engine returns its own identity. Require that, not any failure.
            check_reply(value, batch, identities)
            results.append({"field": name, "request": wrong[name], "observed": value[name], "refused_by_literal_caller": True})
        else:
            raise ValueError("counterparty accepted a wrong requested identity")
    return results
