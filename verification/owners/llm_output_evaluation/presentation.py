# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Versioned presentation diversity, independent of label or model outcomes."""
import json
import re
from data import RATINGS, SCORES, format_pair

VERSION = "source-identity-prose-v1"


def source_prose_view(row):
    group = row.get("group_id")
    if not isinstance(group, str) or re.fullmatch(r"[a-f0-9]{64}", group) is None:
        raise ValueError("prose presentation requires original source group identity")
    context = json.loads(row["prompt"])
    if set(context) != {"task", "output"} or not isinstance(context["output"], str):
        raise ValueError("prose presentation requires complete evaluated output")
    value = dict(row)
    chosen_long = int(group[-8:], 16) % 2 == 0
    for side, key in (("chosen", "label"), ("rejected", "wrong_label")):
        label = row[key]
        if label not in RATINGS:
            raise ValueError("unknown prose rating label")
        score = SCORES[RATINGS.index(label)]
        long = chosen_long if side == "chosen" else not chosen_long
        if long:
            text = f"My overall assessment is {label} (a score of {score} out of 10).\nHere is the complete output I evaluated:\n{context['output']}"
        else:
            text = f"Overall rating: {label}.\nScore: {score} out of 10.\nEvaluated output: {context['output']}"
        value[side] = text
        value[side + "_presentation"] = "prose_narrative" if long else "prose_lines"
    value.update(format="prose", presentation_version=VERSION)
    return value


def evaluation_view(row, name, recipe):
    if recipe == "format-transfer-v3" and name == "prose":
        return source_prose_view(row)
    return format_pair(row, name)
