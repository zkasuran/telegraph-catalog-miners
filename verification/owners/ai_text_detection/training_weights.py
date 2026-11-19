# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pure bounded train-only format-weight policy, not a score or acceptance gate."""
import math
from collections import defaultdict


def weight_policy(rows, bare_weight):
    if type(bare_weight) not in (int, float) or not math.isfinite(bare_weight) or not 1 <= bare_weight <= 32:
        raise ValueError("bounded bare training weight required")
    groups = defaultdict(list)
    weights = []
    for row in rows:
        if row["split"] != "train" or row["style"] not in {"json", "prose", "bare_label"}:
            raise ValueError("format weighting may use training pairs only")
        weight = float(bare_weight if row["style"] == "bare_label" else 1)
        weights.append(weight)
        groups[row["view_id"]].append((row, weight))
    for pairs in groups.values():
        if len(pairs) != 2 or {r["origin"] for r, _ in pairs} != {"human_control", "synthetic"} or len({w for _, w in pairs}) != 1:
            raise ValueError("training weight must be symmetric within each source view")
    if not weights:
        raise ValueError("no training weights")
    total = sum(weights)
    label_totals = {label: sum(w for r, w in zip(rows, weights) if r["label"] == label) for label in ("Human", "AI-generated")}
    if label_totals["Human"] != label_totals["AI-generated"]:
        raise ValueError("training format weights introduced an origin prior")
    return weights, {"rows": len(rows), "bare_weight": float(bare_weight), "mean_training_weight": total / len(weights),
                     "total_training_weight": total, "weight_by_origin": label_totals,
                     "weighting_scope": "Training loss only, not validation/audit metrics or selection gates.",
                     "acceptance_thresholds_changed": False}
