# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Exact native/reference tokenizer parity, with no neural imports."""
VOCAB_SIZE = 50368
CLS, SEP = 50281, 50282
MAX_TEXT_BYTES = 1 << 20


def checked_tokens(text, native_tokenizer, node_tokenizer, max_tokens=256):
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ValueError("rendered text must be a bounded UTF-8 string")
    if type(max_tokens) is not int or not 2 <= max_tokens <= 512:
        raise ValueError("invalid fixed token cap")
    native = native_tokenizer(text, truncation=False, padding=False, add_special_tokens=True)["input_ids"]
    node = node_tokenizer.encode(text, add_special_tokens=True).ids
    for ids in (native, node):
        if not isinstance(ids, list) or not 2 <= len(ids) <= max_tokens:
            raise ValueError("whole rendered input exceeds the fixed token cap")
        if any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in ids):
            raise ValueError("invalid tokenizer token id")
        if ids[0] != CLS or ids[-1] != SEP:
            raise ValueError("canonical CLS and closing SEP must be preserved")
    if native != node:
        raise ValueError("native tokenizer differs from pinned node tokenizer")
    return list(node)
