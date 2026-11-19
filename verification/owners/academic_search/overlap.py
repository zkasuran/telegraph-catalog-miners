# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Vocabulary-free shortcuts include unseen terms. Fit direction on train only."""
import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

from data import FORMATS, words
from source import sha, write


def signals(query, answer):
    q, a = Counter(words(query)), Counter(words(answer))
    shared = len(set(q) & set(a))
    union = len(set(q) | set(a))
    dot = sum(v * a[w] for w, v in q.items())
    norm = sum(v * v for v in q.values()) * sum(v * v for v in a.values())
    return {"literal_shared_words": Fraction(shared), "literal_jaccard": Fraction(shared, max(union, 1)),
            "literal_cosine_squared": Fraction(dot * dot, max(norm, 1))}


def fit(train, training_sha256):
    directions = Counter()
    for row in train:
        a, b = signals(row["prompt"], row["chosen_title"]), signals(row["prompt"], row["rejected_title"])
        for name in a:
            directions[name] += (a[name] > b[name]) - (a[name] < b[name])
    return {"schema_version": 1, "fit_split": "train", "training_sha256": training_sha256,
            "orientations": {k: 1 if v >= 0 else -1 for k, v in directions.items()},
            "coefficient_arithmetic": "integer pair-direction counts; exact rational replay", "audit_used_for_fit": False}


def replay(fixed, rows, training_sha256, format_name):
    if fixed["training_sha256"] != training_sha256 or fixed["fit_split"] != "train" or fixed["audit_used_for_fit"] is not False:
        raise ValueError("unseen-term control training identity mismatch")
    expected = {"literal_shared_words", "literal_jaccard", "literal_cosine_squared"}
    if set(fixed["orientations"]) != expected or any(type(v) is not int or v not in (-1, 1) for v in fixed["orientations"].values()):
        raise ValueError("invalid train-fitted overlap directions")
    totals = Counter()
    for row in rows:
        a, b = (signals(row["prompt"], row["formats"][format_name][side]) for side in ("chosen", "rejected"))
        for name, direction in fixed["orientations"].items():
            score = direction * (a[name] - b[name])
            totals[name] += 2 if score > 0 else 1 if score == 0 else 0
    return {name: total / (2 * len(rows)) for name, total in totals.items()}


def development(root):
    manifest = json.loads((root / "freeze.json").read_bytes())
    raw = (root / "train.json").read_bytes()
    if sha(raw) != manifest["splits"]["train"]["sha256"]:
        raise ValueError("training source changed")
    fixed_path = root / "unseen-term-baselines.json"
    if not fixed_path.exists():
        write(fixed_path, fit(json.loads(raw), sha(raw)))
    fixed = json.loads(fixed_path.read_bytes())
    metrics = {}
    for split in ("train", "validation"):
        payload = (root / (split + ".json")).read_bytes()
        if sha(payload) != manifest["splits"][split]["sha256"]:
            raise ValueError("source split bytes changed")
        rows = json.loads(payload)
        metrics[split] = {f: replay(fixed, rows, sha(raw), f) for f in FORMATS}
    prior = json.loads((root / "development-controls.json").read_bytes())
    failures = prior["failures"] + [f + "/" + c + " vocabulary-free validation shortcut above 0.65"
                                   for f, values in metrics["validation"].items() for c, v in values.items() if v > .65]
    report = {"schema_version": 1, "status": "DEVELOPMENT_SHORTCUT_FAIL" if failures else "DEVELOPMENT_SHORTCUT_PASS_REMOTE_GATE_PENDING",
              "training_allowed": not failures, "unseen_term_controls_sha256": sha(fixed_path.read_bytes()),
              "training_sha256": sha(raw), "metrics": metrics, "failures": failures,
              "prior_controls_sha256": sha((root / "development-controls.json").read_bytes()),
              "audit_scored": False, "audit_used_for_selection": False, "full_primary_dataset_gate_run": False}
    write(root / "development-controls-v2.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-root", type=Path, required=True)
    result = development(parser.parse_args().frozen_root)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["training_allowed"] else 1)
