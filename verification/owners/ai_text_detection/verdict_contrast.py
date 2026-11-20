# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Same-verdict source contrast on existing paired score tensors, no new rows."""
import json
import math
import re


def contrast_policy(recipe):
    expected = {"coefficient": 0.25, "scope": "existing_paired_training_views", "same_verdict_required": True,
                "new_rows": 0, "runtime_transform": False, "source_labels_changed": False, "audit_used": False}
    spec = recipe.get("same_verdict_contrast")
    if (spec != expected or type(spec["coefficient"]) not in (int, float) or type(spec["new_rows"]) is not int or spec["same_verdict_required"] is not True or
            any(spec[k] is not False for k in ("runtime_transform", "source_labels_changed", "audit_used"))):
        raise ValueError("fixed same-verdict source-contrast policy required")
    return dict(expected)


def verdict(row, side):
    text = row[side]
    if row["style"] == "bare_label":
        return text
    if row["style"] == "json":
        return json.loads(text)["verdict"]
    match = re.search(r"Verdict: (Human|AI-generated)(?:\.|$)", text)
    if match is None:
        raise ValueError("literal training verdict could not be read")
    return match.group(1)


def check_pair(rows):
    if (len(rows) != 2 or any(r["split"] != "train" for r in rows) or len({r["view_id"] for r in rows}) != 1 or
            {r["origin"] for r in rows} != {"human_control", "synthetic"} or len({r["style"] for r in rows}) != 1):
        raise ValueError("same-verdict contrast needs one original paired training view")
    for i in range(2):
        if verdict(rows[i], "chosen") != verdict(rows[1-i], "rejected") or verdict(rows[i], "chosen") != rows[i]["label"]:
            raise ValueError("cross-source compared answers do not have the same literal verdict")


def scalar_contrast(chosen, rejected):
    if len(chosen) != 2 or len(rejected) != 2 or any(not math.isfinite(v) for v in chosen + rejected):
        raise ValueError("two finite source score controls required")
    differences = [rejected[1-i] - chosen[i] for i in range(2)]
    return sum(max(v, 0) + math.log1p(math.exp(-abs(v))) for v in differences) / 2
