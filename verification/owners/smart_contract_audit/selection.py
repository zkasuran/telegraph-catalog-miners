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
            "aggregate_accuracy": float(Fraction(sum(wins), len(wins))), "formats": formats,
            "audit_used_for_selection": False, "registration_ready": False,
            "limits": "Native validation selection only. Actual exported-engine validation and independent audit remain separate unchanged gates."}
