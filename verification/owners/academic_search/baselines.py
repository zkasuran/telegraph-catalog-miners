# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Train-fitted integer control coefficients with exact immutable replay."""
import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

from data import words
from source import sha, write


def features(text):
    return [len(text), len(words(text)), text.count("\n"), sum(c.isdigit() for c in text), text.count(":"), text.count("https://")]


def fit(train, training_sha256):
    if len(train) != 128 or len({r["block_id"] for r in train}) != 128:
        raise ValueError("baseline fit requires 128 distinct training originals")
    df, polarity = Counter(), Counter()
    surface = [0] * 6
    for row in train:
        for field in ("query_title", "chosen_title", "rejected_title"):
            df.update(set(words(row[field])))
        polarity.update(set(words(row["chosen_title"])))
        polarity.subtract(set(words(row["rejected_title"])))
        for i, (a, b) in enumerate(zip(features(row["chosen_title"]), features(row["rejected_title"]))):
            surface[i] += (a > b) - (a < b)
    vocabulary = sorted(df, key=lambda term: (-df[term], term))[:2000]
    return {"schema_version": 1, "fit_split": "train", "training_sha256": training_sha256,
            "training_groups": 128, "vocabulary": vocabulary, "document_frequencies": [df[w] for w in vocabulary],
            "answer_only_coefficients": [polarity[w] for w in vocabulary], "surface_coefficients": surface,
            "length_direction": 1 if surface[0] >= 0 else -1,
            "coefficient_arithmetic": "integer source counts, no floating-point fitting", "audit_used_for_fit": False}


def validate(frozen, training_sha256):
    if frozen.get("fit_split") != "train" or frozen.get("training_sha256") != training_sha256 or frozen.get("audit_used_for_fit") is not False:
        raise ValueError("frozen controls are not bound to training only")
    vocab = frozen["vocabulary"]
    if not isinstance(vocab, list) or not 1 <= len(vocab) <= 2000 or len(set(vocab)) != len(vocab):
        raise ValueError("frozen vocabulary shape")
    for name in ("document_frequencies", "answer_only_coefficients"):
        if len(frozen[name]) != len(vocab) or any(type(v) is not int for v in frozen[name]):
            raise ValueError("frozen coefficient dimensions/types")
    if any(not 1 <= v <= 384 for v in frozen["document_frequencies"]):
        raise ValueError("frozen document frequency bounds")
    if len(frozen["surface_coefficients"]) != 6 or any(type(v) is not int for v in frozen["surface_coefficients"]):
        raise ValueError("frozen surface coefficients")
    if type(frozen["length_direction"]) is not int or frozen["length_direction"] not in (-1, 1):
        raise ValueError("invalid train-fitted length direction")


def replay(frozen, rows, training_sha256):
    validate(frozen, training_sha256)
    if not rows:
        raise ValueError("cannot evaluate empty controls")
    vocabulary = tuple(frozen["vocabulary"])
    df = tuple(frozen["document_frequencies"])
    coefficients = tuple(frozen["answer_only_coefficients"])
    style = tuple(frozen["surface_coefficients"])
    totals = Counter()
    for row in rows:
        q = set(words(row["query_title"]))
        a, b = set(words(row["chosen_title"])), set(words(row["rejected_title"]))
        overlap = sum((Fraction(int(w in a) - int(w in b), d) for w, d in zip(vocabulary, df) if w in q), Fraction())
        lexical = sum(c * (int(w in a) - int(w in b)) for w, c in zip(vocabulary, coefficients))
        surface = sum(c * (x - y) for c, x, y in zip(style, features(row["chosen_title"]), features(row["rejected_title"])))
        length = frozen["length_direction"] * (len(row["chosen_title"]) - len(row["rejected_title"]))
        for name, score in (("overlap", overlap), ("answer_only", lexical), ("surface", surface), ("length", length)):
            totals[name] += 2 if score > 0 else 1 if score == 0 else 0
    return {name: value / (2 * len(rows)) for name, value in totals.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-root", type=Path, required=True)
    root = parser.parse_args().frozen_root
    raw = (root / "train.json").read_bytes()
    manifest = json.loads((root / "freeze.json").read_bytes())
    if sha(raw) != manifest["splits"]["train"]["sha256"]:
        raise ValueError("training bytes changed")
    controls = fit(json.loads(raw), sha(raw))
    write(root / "frozen-baselines.json", controls)
    # The command fits exactly once. Validation/audit replays never refit.
    write(root / "baseline-manifest.json", {"fit_split": "train", "training_sha256": sha(raw),
          "frozen_baselines_sha256": sha((root / "frozen-baselines.json").read_bytes()), "audit_scored": False})
