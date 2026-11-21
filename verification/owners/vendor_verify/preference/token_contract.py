# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Full development-token diagnostics with no source text in the result."""

from collections import Counter

import data

MAX_TOKENS = 512
MIN_RETAINED_TRAIN_PAIRS = 64


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


def filter_training(rows, tokenize_full):
    """Exclude whole training pairs solely for exact token capacity, not quality."""
    if not rows or len(rows) > data.MAX_PAIRS:
        raise ValueError("training count ceiling")
    retained, excluded, groups = [], [], set()
    for row in rows:
        group = row.get("group_id")
        if (
            not isinstance(group, str)
            or len(group) != 64
            or any(c not in "0123456789abcdef" for c in group)
            or group in groups
        ):
            raise ValueError("unique normalized training group required")
        groups.add(group)
        lengths = {}
        for side in ("chosen", "rejected"):
            ids = tokenize_full(data.render(row["prompt"], row[side]))
            observed = compare_ids(ids, ids, ids)
            lengths[side] = observed["full_tokens"]
        if max(lengths.values()) > MAX_TOKENS:
            excluded.append(
                {
                    "group_id": group,
                    "row_sha256": data.sha(data.encode(row)),
                    "token_counts": lengths,
                    "reason": "exact_token_capacity",
                }
            )
        else:
            retained.append(row)
    if len(retained) < MIN_RETAINED_TRAIN_PAIRS or len(excluded) * 10 > len(rows):
        raise ValueError("insufficient training coverage after capacity exclusions")
    original = b"".join(data.encode(row) for row in rows)
    encoded = b"".join(data.encode(row) for row in retained)
    report = {
        "schema_version": 1,
        "policy": "whole training pairs over exact 512-token capacity only",
        "original_train_sha256": data.sha(original),
        "training_sha256": data.sha(encoded),
        "original_pairs": len(rows),
        "retained_pairs": len(retained),
        "excluded": excluded,
        "raw_text_exported": False,
        "validation_changed": False,
        "audit_checked": False,
    }
    return retained, encoded, report


def require_training_capacity_only(report):
    """An exclusion cannot repair mismatched tokens or held-out truncation."""
    if (
        report.get("max_tokens") != MAX_TOKENS
        or report.get("audit_checked") is not False
    ):
        raise ValueError("exact development diagnostic required")
    train = report["splits"]["train"]["counts"]
    validation = report["splits"]["validation"]["counts"]
    causes = (
        "over_cap",
        "training_node_mismatch",
        "training_truncated",
        "node_truncated",
    )
    if any(
        type(counts.get(name)) is not int or counts[name] < 0
        for counts in (train, validation)
        for name in causes
    ):
        raise ValueError("actual nonnegative diagnostic counts required")
    if any(validation[name] for name in causes) or train["training_node_mismatch"]:
        raise ValueError("identity or held-out retention failure cannot be filtered")
    if (
        train["training_truncated"] != train["over_cap"]
        or train["node_truncated"] != train["over_cap"]
    ):
        raise ValueError("noncapacity token loss cannot be filtered")
    return True
