# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Validation-only gate-margin selection. No model imports or audit inputs."""
import math
from fractions import Fraction

FORMATS = ("prose", "json", "report")
MECHANISMS = ("access_control", "arithmetic")


def selection_metrics(rows, differences):
    if not 0 < len(rows) <= 768 or len(rows) != len(differences):
        raise ValueError("aligned bounded validation differences required")
    for row, difference in zip(rows, differences):
        if row.get("split") != "validation":
            raise ValueError("selection accepts validation only, never train or audit")
        if row.get("response_view") not in FORMATS or type(row.get("vulnerable")) is not bool or row.get("facts", {}).get("mechanism") not in MECHANISMS:
            raise ValueError("complete format/label/mechanism identity required")
        if type(difference) not in (int, float) or not math.isfinite(difference):
            raise ValueError("finite real validation differences required")
    wins = [difference > 0 for difference in differences]
    margins, formats = [], {}

    def rate(indices):
        if not indices:
            raise ValueError("missing supported validation cell")
        return Fraction(sum(wins[i] for i in indices), len(indices))

    for name in FORMATS:
        indices = [i for i, row in enumerate(rows) if row["response_view"] == name]
        overall = rate(indices)
        labels = {str(value): rate([i for i in indices if rows[i]["vulnerable"] == value]) for value in (True, False)}
        mechanisms = {value: rate([i for i in indices if rows[i]["facts"]["mechanism"] == value]) for value in MECHANISMS}
        margins.append(overall - Fraction(4, 5))
        margins.extend(value - Fraction(7, 10) for value in (*labels.values(), *mechanisms.values()))
        formats[name] = {"accuracy": float(overall), "per_label": {k: float(v) for k, v in labels.items()},
                         "per_mechanism": {k: float(v) for k, v in mechanisms.items()}, "pairs": len(indices)}
    return {"method": "validation worst format/label/mechanism gate margin", "selection_score": float(min(margins)),
            "gate_margins": [[value.numerator, value.denominator] for value in sorted(margins)],
            "correct_pairs": sum(wins), "measured_pairs": len(wins),
            "aggregate_accuracy": float(Fraction(sum(wins), len(wins))), "formats": formats,
            "audit_used_for_selection": False, "registration_ready": False,
            "limits": "Native validation selection only. Actual exported-engine validation and independent audit remain separate unchanged gates."}


def checkpoint_key(result):
    if result.get("audit_used_for_selection") is not False or result.get("registration_ready") is not False:
        raise ValueError("validation-only checkpoint record required")
    margins = result.get("gate_margins")
    if not isinstance(margins, list) or len(margins) != 15:
        raise ValueError("all supported format/label/mechanism margins required")
    exact = []
    for value in margins:
        if not isinstance(value, list) or len(value) != 2 or any(type(v) is not int for v in value) or not -7680 <= value[0] <= 7680 or not 1 <= value[1] <= 7680:
            raise ValueError("bounded exact rational gate margins required")
        margin = Fraction(*value)
        if not -1 <= margin <= 1:
            raise ValueError("validation gate margin outside probability bounds")
        exact.append(margin)
    correct, measured = result.get("correct_pairs"), result.get("measured_pairs")
    if type(correct) is not int or type(measured) is not int or not 0 <= correct <= measured <= 768 or measured == 0:
        raise ValueError("actual bounded validation counts required")
    return tuple(sorted(exact)), Fraction(correct, measured)


def checkpoint_better(result, previous):
    candidate = checkpoint_key(result)
    return previous is None or candidate > checkpoint_key(previous)
