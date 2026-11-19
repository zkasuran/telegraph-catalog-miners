# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Validation acceptance before the independent audit is scored."""
import math


def metrics(rows, scores):
    if len(rows) != 32 or len(scores) != 64 or len({r["group_id"] for r in rows}) != 16:
        raise ValueError("complete original validation groups required")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in scores):
        raise ValueError("finite actual validation scores required")
    values = {"Yes": [], "No": []}
    for index, row in enumerate(rows):
        label = "Yes" if row["variant"] == "original_claim" else "No"
        values[label].append(scores[2 * index] > scores[2 * index + 1])
    if any(len(scores) != 16 for scores in values.values()):
        raise ValueError("balanced validation source labels required")
    return {"accuracy": sum(sum(v) for v in values.values()) / 32,
            "per_label": {name: sum(v) / len(v) for name, v in values.items()}}


def accepted(formats):
    if set(formats) != {"prose", "json"}:
        raise ValueError("both validation formats required")
    for value in formats.values():
        numbers = [value["accuracy"], *value["per_label"].values()]
        if set(value["per_label"]) != {"Yes", "No"} or any(not math.isfinite(n) or not 0 <= n <= 1 for n in numbers):
            raise ValueError("valid per-label validation report required")
        if value["accuracy"] < .80 or min(value["per_label"].values()) < .70:
            return False
    return True
