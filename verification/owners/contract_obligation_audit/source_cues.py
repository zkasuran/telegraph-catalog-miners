# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Frozen source-aware semantic baselines, separate from answer-only tells."""
import re


def counts(row):
    text = row["evidence"].casefold()
    return {"negation": len(re.findall(r"\bnot\b|\bno\b|\bneither\b|\bwithout\b", text)),
            "exception": len(re.findall(r"\bexcept\b|\bunless\b|\bprovided\b|\bsubject\b", text))}


def measure_source_cues(rows, frozen):
    if frozen["fit_split"] != "train" or set(frozen["integer_coefficients"]) != {"negation", "exception"}:
        raise ValueError("source-aware baseline must be an immutable training-only snapshot")
    result = {}
    for name, params in frozen["integer_coefficients"].items():
        if type(params["threshold"]) is not int or params["direction"] not in (1, -1):
            raise ValueError("integer semantic baseline coefficients required")
        label = "Entailment" if params["direction"] == 1 else "Contradiction"
        result[name] = sum((counts(row)[name] >= params["threshold"]) == (row["label"] == label) for row in rows)/len(rows)
    return result
