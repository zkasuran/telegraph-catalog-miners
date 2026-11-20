# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pure source-view coupling, preserving the exact training pair population."""
import math
from collections import defaultdict


def coupled_policy(recipe):
    expected = {"gap_coefficient": 0.25, "rows_per_view": 2, "origin_order": "role_symmetric",
                "row_accumulation": 8, "runtime_transform": False}
    if recipe.get("coupled_objective") != expected:
        raise ValueError("fixed symmetric counterpart objective required")
    return expected.copy()


def source_groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] != "train":
            raise ValueError("source-view objective may fit training only")
        grouped[row["view_id"]].append(row)
    if not rows:
        raise ValueError("empty source-view training")
    groups = []
    for key in sorted(grouped):
        group = grouped[key]
        if (len(group) != 2 or {r["origin"] for r in group} != {"human_control", "synthetic"} or
                any(len({r[k] for r in group}) != 1 for k in ("source_id", "group_id", "style", "split", "year")) or
                len({r["text_id"] for r in group}) != 2):
            raise ValueError("counterpart views must retain balanced source identity and format")
        groups.append(sorted(group, key=lambda r: r["origin"]))
    return groups


def scalar_coupled(a, b):
    if any(not math.isfinite(x) or x < 0 for x in (a, b)):
        raise ValueError("finite nonnegative counterpart losses required")
    return (a + b) / 2 + 0.25 * abs(a - b)
