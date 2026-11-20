# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Immutable train-fitted integer group sampling weights, no neural imports."""
import argparse
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

IDENTITY_FIELDS = ("id", "source_sha256", "template_group", "response_view")


def identity(rows):
    if not 0 < len(rows) <= 768:
        raise ValueError("bounded nonempty training views required")
    for row in rows:
        if row.get("split") != "train" or row.get("response_view") not in ("prose", "json", "report"):
            raise ValueError("sampling accepts training views only")
        if any(not isinstance(row.get(key), str) or not row[key] for key in IDENTITY_FIELDS):
            raise ValueError("complete training view identity required")
    return hashlib.sha256(json.dumps([{key: r[key] for key in IDENTITY_FIELDS} for r in rows], sort_keys=True).encode()).hexdigest()


def fit(rows):
    view_identity = identity(rows)
    counts = Counter(r["template_group"] for r in rows)
    maximum = max(counts.values())
    if any(maximum % count or maximum // count > 16 for count in counts.values()):
        raise ValueError("exact bounded integer group weights required")
    return {"schema_version": 1, "SPDX-License-Identifier": "LicenseRef-zkasuran-SAND-1.0",
            "method": "train-only inverse implementation view count, frozen integers",
            "view_identity": view_identity, "group_view_counts": dict(sorted(counts.items())),
            "coefficients": {group: maximum // count for group, count in sorted(counts.items())},
            "replacement": True, "samples_per_epoch": len(rows), "audit_used_for_fitting": False}


def weights(rows, frozen):
    if not isinstance(frozen, dict) or frozen.get("schema_version") != 1 or frozen.get("audit_used_for_fitting") is not False:
        raise ValueError("frozen train-only sampling record required")
    if frozen.get("view_identity") != identity(rows) or frozen.get("samples_per_epoch") != len(rows) or frozen.get("replacement") is not True:
        raise ValueError("sampling training identity changed")
    counts = Counter(r["template_group"] for r in rows)
    coefficients = frozen.get("coefficients", {})
    if not isinstance(coefficients, dict):
        raise ValueError("frozen coefficient mapping required")
    if dict(counts) != frozen.get("group_view_counts") or set(counts) != set(coefficients):
        raise ValueError("sampling group population changed")
    if any(type(value) is not int or not 1 <= value <= 16 for value in coefficients.values()):
        raise ValueError("positive bounded frozen integer coefficients required")
    masses = {coefficients[group] * count for group, count in counts.items()}
    if len(masses) != 1:
        raise ValueError("frozen weights do not equalize implementation probability")
    return [coefficients[r["template_group"]] for r in rows]


def main():
    from data import response_views
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    with args.train.open("rb") as stream:
        raw = stream.read((2 << 20) + 1)
    if len(raw) > 2 << 20:
        raise ValueError("training source byte ceiling")
    original = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    rows = response_views(original, "train")
    if args.freeze:
        if args.frozen.exists():
            raise ValueError("frozen sampling coefficients already exist; refusing overwrite")
        args.frozen.write_text(json.dumps(fit(rows), indent=2, sort_keys=True) + "\n")
    frozen = json.loads(args.frozen.read_text())
    actual = weights(rows, frozen)
    counts = Counter(r["template_group"] for r in rows)
    masses = {g: counts[g] * frozen["coefficients"][g] for g in counts}
    report = {"training_originals": len(original), "training_views": len(rows), "training_groups": len(counts),
              "group_view_counts": dict(counts), "coefficients": frozen["coefficients"],
              "weighted_probability_per_group": {g: float(Fraction(m, sum(masses.values()))) for g, m in masses.items()},
              "unweighted_retained_design_probability": float(Fraction(sum(counts[g] for g in counts if g.startswith("design-")), len(rows))),
              "weighted_retained_design_probability": float(Fraction(sum(masses[g] for g in masses if g.startswith("design-")), sum(masses.values()))),
              "samples_per_epoch": len(actual), "replacement": True,
              "frozen_sha256": hashlib.sha256(args.frozen.read_bytes()).hexdigest(),
              "train_source_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
              "model_or_torch_imported": False, "audit_used_for_fitting": False,
              "limits": "Sampling repeats existing training views, not new sources or human annotations. Expected group balance is not measured model improvement."}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
