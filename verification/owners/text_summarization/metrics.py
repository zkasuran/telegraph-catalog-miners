# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Complete reciprocal-group metrics from actual finite scores."""
import math
from collections import Counter

from data import VIEWS


def pair_metrics(rows, scores):
    if not rows or len(scores) != 2 * len(rows) or any(type(x) not in (int, float) or not math.isfinite(x) for x in scores):
        raise ValueError("complete finite pair scores required")
    counts = Counter(r["group_id"] for r in rows)
    if any(n != 2 for n in counts.values()) or Counter(r["target"] for r in rows) != {"A": len(rows) // 2, "B": len(rows) // 2}:
        raise ValueError("complete reciprocal A/B groups required")
    wins = [scores[2 * i] > scores[2 * i + 1] for i in range(len(rows))]
    subsets = {"target_A": [i for i, r in enumerate(rows) if r["target"] == "A"],
               "target_B": [i for i, r in enumerate(rows) if r["target"] == "B"],
               "chosen_longer": [i for i, r in enumerate(rows) if len(r["chosen"]) > len(r["rejected"])],
               "chosen_shorter": [i for i, r in enumerate(rows) if len(r["chosen"]) < len(r["rejected"])]}
    if any(not indices for indices in subsets.values()):
        raise ValueError("missing target or length subset")
    return {"n_eval": len(rows), "independent_groups": len(counts), "correct": sum(wins),
            "pairwise_accuracy": sum(wins) / len(rows),
            "subsets": {name: sum(wins[i] for i in indices) / len(indices) for name, indices in subsets.items()},
            "per_label": {label: sum(wins[i] for i in subsets["target_" + label]) / len(subsets["target_" + label]) for label in ("A", "B")}}


def quality(trained, untrained, controls):
    failures = []
    accuracy = trained["pairwise_accuracy"]
    values = [accuracy, untrained["pairwise_accuracy"], *controls.values(), *trained["subsets"].values()]
    if not controls or not trained["subsets"] or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        return {"passed": False, "failures": ["invalid or missing quality metric"]}
    if accuracy < .80:
        failures.append("audit ordering below 80 percent")
    if accuracy + 1e-9 < untrained["pairwise_accuracy"] + .05:
        failures.append("less than five points above seed-matched untrained")
    if accuracy + 1e-9 < max(controls.values()) + .10:
        failures.append("less than ten points above strongest frozen shortcut")
    if any(value > .65 for value in controls.values()):
        failures.append("audit shortcut exceeds 65 percent")
    if any(value < .70 for value in trained["subsets"].values()):
        failures.append("claimed target or length subset below 70 percent")
    return {"passed": not failures, "failures": failures}


def all_formats(trained, untrained, controls):
    if any(set(value) != set(VIEWS) for value in (trained, untrained, controls)):
        raise ValueError("all claimed formats must actually be measured")
    results = {style: quality(trained[style], untrained[style], controls[style]) for style in VIEWS}
    return {"passed": all(r["passed"] for r in results.values()), "by_format": results}
