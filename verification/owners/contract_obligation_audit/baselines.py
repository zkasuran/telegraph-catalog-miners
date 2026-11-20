# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Immutable train-fitted integer shortcuts, never neural training."""
import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from contracts import HERE, FORMATS, read_jsonl, sha, strict_json, write_json
from data import pair


def words(text):
    return Counter(re.findall(r"[a-z0-9']+", text.casefold()))


def fit(train):
    coefficients = Counter()
    hypotheses = defaultdict(Counter)
    for row in train:
        coefficients.update(words(row["chosen"]))
        coefficients.subtract(words(row["rejected"]))
        hypotheses[row["hypothesis_id"]][row["label"]] += 1
    return {"schema_version": 1, "fit_split": "train", "fit_rows": len(train),
            "train_sha256": sha(b"".join((row["document_id"]+":"+row["hypothesis_id"]+":"+row["evidence_sha256"]+"\n").encode() for row in train)),
            "bow_integer_coefficients": {key: value for key, value in sorted(coefficients.items()) if value},
            "hypothesis_label_counts": {key: dict(value) for key, value in sorted(hypotheses.items())},
            "length_direction": 1, "overlap_direction": 1}


def validate_frozen(frozen, train):
    # Deterministic integer equality validates a fixed snapshot. Scoring never
    # fits a fresh floating-point model and never reads validation/audit labels.
    if frozen != fit(train):
        raise ValueError("frozen baseline does not bind the exact training originals")


def differences(row, frozen):
    chosen, rejected = row["chosen"], row["rejected"]
    bow = frozen["bow_integer_coefficients"]
    countc, countr, prompt = words(chosen), words(rejected), words(row["prompt"])
    prior = frozen["hypothesis_label_counts"].get(row["hypothesis_id"], {})
    expected = "Entailment" if prior.get("Entailment", 0) >= prior.get("Contradiction", 0) else "Contradiction"
    return {"length": len(chosen)-len(rejected),
            "unigram_overlap": sum((countc & prompt).values())-sum((countr & prompt).values()),
            "answer_only_bow": sum((countc[w]-countr[w])*n for w, n in bow.items()),
            "exact_quote": int(row["evidence"] in chosen)-int(row["evidence"] in rejected),
            "hypothesis_prior": 1 if row["label"] == expected else -1}


def measure(rows, frozen):
    diff = [differences(row, frozen) for row in rows]
    return {key: sum(.5 if value[key] == 0 else int(value[key] > 0) for value in diff)/len(diff) for key in diff[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    train = read_jsonl(args.data / "train.jsonl")
    path = HERE / "frozen-baselines-v8.json"
    if args.freeze:
        if path.exists():
            raise ValueError("frozen baseline already exists; do not overwrite it")
        write_json(path, fit(train))
    frozen = strict_json(path.read_bytes())
    validate_frozen(frozen, train)
    # Never expose the reserved audit while developing a candidate.
    report = {name: {style: measure([pair(row, style) for row in read_jsonl(args.data / (name+".jsonl"))], frozen)
                     for style in FORMATS} for name in ("train", "validation")}
    write_json(args.data / "shortcut-report.json", report)
    print(report)


if __name__ == "__main__":
    main()
