# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Literal public request/response contract, no neural imports."""
from contracts import FOUNDATION, PACKAGE_KECCAK


def request_ok(value):
    if not isinstance(value, dict) or not isinstance(value.get("batch"), list) or not 1 <= len(value["batch"]) <= 100:
        raise ValueError("bounded nonempty batch required")
    expected = {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"], "tokenizer_hash": FOUNDATION["tokenizer.json"], "adapter_hash": PACKAGE_KECCAK}
    if any(value.get(key) != identity for key, identity in expected.items()):
        raise ValueError("literal bare runtime identities required")
    for item in value["batch"]:
        tokens = item.get("tokens") if isinstance(item, dict) else None
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= 256 or any(type(t) is not int or not 0 <= t < 50368 for t in tokens):
            raise ValueError("token budget/range refused; never truncate")
    return value


def response_ok(value, count):
    if not isinstance(value, dict) or value.get("adapter_hash") != PACKAGE_KECCAK or value.get("foundation_hash") != FOUNDATION["foundation_lora_fp32.onnx"] or value.get("tokenizer_hash") != FOUNDATION["tokenizer.json"]:
        raise ValueError("executor did not return exact identities")
    results = value.get("results")
    if not isinstance(results, list) or len(results) != count:
        raise ValueError("wrong executor result count")
    for item in results:
        bit = item.get("score") if isinstance(item, dict) else None
        if type(bit) is not int or not 0 <= bit < 1 << 32 or bit & 0x7f800000 == 0x7f800000:
            raise ValueError("nonfinite or invalid score bits")
    return value
