# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Aggregate development attribution. Displaced prompts are not new truth labels."""

import math

import data

RIDGE_GRID = (0.1, 1.0, 10.0, 100.0)


def variants(rows):
    if not rows or len(rows) > 10000:
        raise ValueError("bounded nonempty development pairs required")
    for row in rows:
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in data.FIELDS):
            raise ValueError("invalid development text")
    return {
        "literal": rows,
        "answer_only": [{**row, "prompt": ""} for row in rows],
        "displaced_prompt": [
            {**row, "prompt": rows[(i + 1) % len(rows)]["prompt"]}
            for i, row in enumerate(rows)
        ],
        "prompt_only": [{**row, "chosen": "", "rejected": ""} for row in rows],
    }


def aggregate(rows, diffs):
    if not rows or len(rows) != len(diffs) or len(rows) > 10000:
        raise ValueError("complete bounded development score coverage required")
    if any(type(v) not in {int, float} or not math.isfinite(v) for v in diffs):
        raise ValueError("finite literal score differences required")

    def group(indices):
        values = [diffs[i] for i in indices]
        return {
            "pairs": len(values),
            "accuracy": sum(v > 0 for v in values) / len(values) if values else None,
            "ties": sum(v == 0 for v in values),
            "mean_difference": math.fsum(values) / len(values) if values else None,
        }

    result = group(range(len(rows)))
    result["preferred_response"] = {
        str(label): group(
            [
                i
                for i, row in enumerate(rows)
                if row.get("source_preferred_response") == label
            ]
        )
        for label in (1, 2)
    }
    result["length_direction"] = {
        name: group(
            [
                i
                for i, row in enumerate(rows)
                if (len(row["chosen"]) - len(row["rejected"]) > 0) == positive
            ]
        )
        for name, positive in (("chosen_longer", True), ("chosen_not_longer", False))
    }
    result["source_strength"] = {
        name: group(
            [
                i
                for i, row in enumerate(rows)
                if (abs(row.get("source_strength", 0)) >= 2) == strong
            ]
        )
        for name, strong in (("at_least_two", True), ("less_than_two", False))
    }
    return result


def select_candidate(candidates):
    if (
        len(candidates) != len(RIDGE_GRID)
        or tuple(c["ridge"] for c in candidates) != RIDGE_GRID
    ):
        raise ValueError("predeclared complete ridge grid required")
    for candidate in candidates:
        metric = candidate.get("validation", {})
        if (
            type(metric.get("pairs")) is not int
            or metric["pairs"] < 1
            or type(metric.get("accuracy")) not in {int, float}
            or not math.isfinite(metric["accuracy"])
            or not 0 <= metric["accuracy"] <= 1
        ):
            raise ValueError("actual finite validation result required")
    # Prefer stronger regularization on exact validation ties, never audit scores.
    return max(
        range(len(candidates)),
        key=lambda i: (candidates[i]["validation"]["accuracy"], candidates[i]["ridge"]),
    )


def qualification_development_gate(metric, untrained, strongest_shortcut):
    if not metric["pairs"] or metric["accuracy"] is None:
        return False
    labels = list(metric["preferred_response"].values())
    return (
        metric["accuracy"] >= 0.8
        and metric["accuracy"] + 1e-9 >= untrained + 0.05
        and metric["accuracy"] + 1e-9 >= strongest_shortcut + 0.10
        and all(
            m["pairs"] > 0 and m["accuracy"] is not None and m["accuracy"] >= 0.7
            for m in labels
        )
    )
