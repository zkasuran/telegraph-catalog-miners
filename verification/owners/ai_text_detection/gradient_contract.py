# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pure fixed-view selection and gradient-geometry controls, no neural imports."""
import math

from common import FORMATS, identity
from verdict_contrast import check_pair


def diagnostic_policy(recipe):
    expected = {"scope": "training_only", "views_per_format": 2, "maximum_views": 6,
                "selection": "lowest_group_and_view_hashes_without_scores", "time_limit_seconds": 900,
                "optimizer_updates": 0, "validation_used": False, "audit_used": False,
                "gradient_surgery": False, "registration_eligible": False}
    spec = recipe.get("gradient_diagnostic")
    if (spec != expected or any(type(spec[k]) is not int for k in ("views_per_format", "maximum_views", "time_limit_seconds", "optimizer_updates")) or
            any(spec[k] is not False for k in ("validation_used", "audit_used", "gradient_surgery", "registration_eligible"))):
        raise ValueError("fixed training-only no-update diagnostic required")
    return dict(expected)


def select_views(rows, policy):
    if len(rows) != 1008 or any(r["split"] != "train" for r in rows):
        raise ValueError("gradient diagnostic requires the exact complete training population")
    groups = {}
    for row in rows:
        if row["group_id"] != identity(row["source_id"]):
            raise ValueError("gradient diagnostic source-group identity changed")
        groups.setdefault(row["view_id"], []).append(row)
    for group in groups.values():
        check_pair(group)
    result, used = [], set()
    for style in FORMATS:
        candidates = sorted((g for g in groups.values() if g[0]["style"] == style), key=lambda g: (g[0]["group_id"], g[0]["view_id"]))
        selected = 0
        for group in candidates:
            if group[0]["group_id"] in used:
                continue
            result.append(group)
            used.add(group[0]["group_id"])
            selected += 1
            if selected == policy["views_per_format"]:
                break
        if selected != policy["views_per_format"]:
            raise ValueError("fixed diagnostic lacks enough independent format groups")
    if len(result) != policy["maximum_views"]:
        raise ValueError("gradient diagnostic view ceiling changed")
    return result


def geometry(left_squared, right_squared, dot):
    if any(not math.isfinite(v) for v in (left_squared, right_squared, dot)) or min(left_squared, right_squared) < 0:
        raise ValueError("finite nonnegative gradient norms required")
    left, right = math.sqrt(left_squared), math.sqrt(right_squared)
    if left == 0 or right == 0:
        if dot != 0:
            raise ValueError("zero norm cannot have a nonzero gradient dot product")
        return {"left_norm": left, "right_norm": right, "dot": dot, "cosine": None, "opposed": False}
    cosine = dot / (left * right)
    if not math.isfinite(cosine) or abs(cosine) > 1 + 1e-9:
        raise ValueError("gradient geometry violates Cauchy-Schwarz")
    return {"left_norm": left, "right_norm": right, "dot": dot,
            "cosine": max(-1.0, min(1.0, cosine)), "opposed": dot < 0}
