# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Self-reported child metrics from existing frozen audit scores, not qualification."""
import math
import re

FAMILY = "CODE_PATCH_VERIFY"
CHILD = "CODE_PATCH_VERIFY_GENERAL"
MAX_PAIRS = 100000


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + 1 + stop) / 2
        for position in range(start, stop):
            ranks[order[position]] = rank
        start = stop
    return ranks


def measured_training_record(pairs, dataset_sha256, *, audit_used_for_selection):
    """No source truth or audit independence is inferred from a valid record."""
    if not isinstance(dataset_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", dataset_sha256):
        raise ValueError("actual frozen dataset identity required")
    if audit_used_for_selection is not False:
        raise ValueError("frozen independent selection boundary required")
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= MAX_PAIRS:
        raise ValueError("bounded previously measured score pairs required")
    scores = []
    wins = 0
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("two finite native measured rewards required")
        for value in pair:
            try:
                valid = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError("two finite native measured rewards required")
        scores.extend(pair)
        wins += pair[0] > pair[1]
    labels = [value for _ in pairs for value in (1, 0)]
    x, y = average_ranks(scores), average_ranks(labels)
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    dx, dy = [v - mean_x for v in x], [v - mean_y for v in y]
    denominator = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if not denominator:
        raise ValueError("undefined all-tied score correlation; do not invent Spearman")
    spearman = sum(a * b for a, b in zip(dx, dy)) / denominator
    if not math.isfinite(spearman) or not -1.0000000001 <= spearman <= 1.0000000001:
        raise ValueError("invalid measured rank correlation")
    spearman = max(-1.0, min(1.0, spearman))
    return {"family": FAMILY, "children": [{"intent": CHILD, "dataset_version": dataset_sha256,
                                             "pairs": len(pairs), "accuracy": wins / len(pairs),
                                             "spearman": spearman, "held_out": True}]}
