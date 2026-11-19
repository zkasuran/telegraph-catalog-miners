# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Literal reference-engine wire checks. Hash prefixes are never normalized."""
import json
import math
import struct
import urllib.request


def score_value(bits):
    if type(bits) is not int or not 0 <= bits <= 0xffffffff:
        raise ValueError("invalid score bits")
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    if not math.isfinite(value):
        raise ValueError("nonfinite score bits")
    return value


def check_reply(reply, count, identity):
    if not isinstance(reply, dict) or any(reply.get(k) != v for k, v in identity.items()):
        raise ValueError("literal identity echo mismatch")
    results = reply.get("results")
    if not isinstance(results, list) or len(results) != count:
        raise ValueError("wrong batch result count")
    bits = []
    for r in results:
        if not isinstance(r, dict) or set(r) != {"score"}:
            raise ValueError("score or refusal result malformed")
        score_value(r["score"])
        bits.append(r["score"])
    return bits


def request(url, batch, identity):
    raw = json.dumps({"batch": [{"tokens": ids} for ids in batch], **identity}, allow_nan=False).encode()
    if len(raw) > 1 << 20 or not 1 <= len(batch) <= 100:
        raise ValueError("request ceiling")
    req = urllib.request.Request(url, data=raw, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        data = response.read((1 << 20) + 1)
        if response.status != 200 or len(data) > 1 << 20:
            raise ValueError("response status/byte ceiling")
    return json.loads(data)
