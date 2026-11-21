# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Full development-token diagnostics with no source text in the result."""

from collections import Counter

import data

MAX_TOKENS = 512


def compare_ids(full, training, node):
    for values in (full, training, node):
        if (
            not isinstance(values, list)
            or not values
            or any(type(v) is not int or v < 0 for v in values)
        ):
            raise ValueError("invalid token vector")
    return {
        "full_tokens": len(full),
        "training_tokens": len(training),
        "node_tokens": len(node),
        "over_cap": len(full) > MAX_TOKENS,
        "training_node_mismatch": training != node,
        "training_truncated": full != training,
        "node_truncated": full != node,
    }


def inspect_development(splits, tokenize_full, tokenize_training, tokenize_node):
    if set(splits) != {"train", "validation"}:
        raise ValueError("only development splits permitted")
    result = {
        "schema_version": 1,
        "max_tokens": MAX_TOKENS,
        "audit_checked": False,
        "raw_text_exported": False,
        "splits": {},
    }
    totals = Counter()
    for split in ("train", "validation"):
        rows = splits[split]
        if not rows or len(rows) > 10000:
            raise ValueError("development count ceiling")
        counts, maxima = Counter(), Counter()
        for row in rows:
            for side in ("chosen", "rejected"):
                text = data.render(row["prompt"], row[side])
                observed = compare_ids(
                    tokenize_full(text), tokenize_training(text), tokenize_node(text)
                )
                counts["texts_checked"] += 1
                for name in (
                    "over_cap",
                    "training_node_mismatch",
                    "training_truncated",
                    "node_truncated",
                ):
                    counts[name] += int(observed[name])
                for name in ("full_tokens", "training_tokens", "node_tokens"):
                    maxima[name] = max(maxima[name], observed[name])
        result["splits"][split] = {
            "pairs": len(rows),
            "counts": dict(counts),
            "maxima": dict(maxima),
        }
        totals.update(counts)
    result["totals"] = dict(totals)
    result["passed"] = not any(
        totals[k]
        for k in (
            "over_cap",
            "training_node_mismatch",
            "training_truncated",
            "node_truncated",
        )
    )
    return result
