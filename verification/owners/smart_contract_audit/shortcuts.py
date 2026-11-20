# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Integer/fraction train-fitted baselines, frozen before model selection."""
import argparse
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

from data import normalize, sha, write


def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def fit(train):
    chosen, rejected = Counter(), Counter()
    for row in train:
        chosen.update(words(row["chosen"]))
        rejected.update(words(row["rejected"]))
    coefficients = {w: chosen[w] - rejected[w] for w in sorted(chosen.keys() | rejected.keys())}
    # Every answer exists in both directions in a vulnerable/fixed sibling pair.
    # These exact integer coefficients reproduce without a floating-point refit.
    return {"schema_version": 1, "method": "train-only answer token count difference",
            "coefficients": coefficients, "length_direction": direction(train, length),
            "overlap_direction": direction(train, overlap),
            "train_identity": sha(json.dumps(train, sort_keys=True).encode())}


def length(row, side):
    return len(row[side])


def overlap(row, side):
    a, b = words(row["source"]), words(row[side])
    return Fraction(len(a & b), max(1, len(b)))


def accuracy(rows, scorer):
    points = Fraction(0)
    for row in rows:
        a, b = scorer(row, "chosen"), scorer(row, "rejected")
        points += 1 if a > b else Fraction(1, 2) if a == b else 0
    return float(points / len(rows))


def direction(rows, scorer):
    return 1 if accuracy(rows, scorer) >= 0.5 else -1


def evaluate(rows, frozen):
    c = frozen["coefficients"]
    report = {}
    for split in ("train", "validation", "audit"):
        subset = [r for r in rows if r["split"] == split]
        report[split] = {"pairs": len(subset),
                         "length": accuracy(subset, lambda r, s: frozen["length_direction"] * length(r, s)),
                         "source_answer_overlap": accuracy(subset, lambda r, s: frozen["overlap_direction"] * overlap(r, s)),
                         "answer_only": accuracy(subset, lambda r, s: sum(c.get(w, 0) for w in words(r[s])))}
    # Record near-overlap as well as exact normalized equality.
    grouped = {split: {normalize(r["source"]) for r in rows if r["split"] == split} for split in report}
    report["max_cross_split_token_jaccard"] = max(
        float(Fraction(len(words(a) & words(b)), len(words(a) | words(b))))
        for left in grouped for right in grouped if left < right for a in grouped[left] for b in grouped[right])
    report["limits"] = f"{len({r['template_group'] for r in rows})} grouped source implementations, not {len(rows)} independent human annotations. Identical token sets can have different execution paths. Tie baselines score one half; high lexical overlap is disclosed."
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", required=True, type=Path)
    p.add_argument("--frozen", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--freeze", action="store_true")
    args = p.parse_args()
    rows = json.loads(args.inventory.read_text())
    if args.freeze:
        if args.frozen.exists():
            raise ValueError("frozen coefficients already exist; refusing to overwrite")
        write(args.frozen, fit([r for r in rows if r["split"] == "train"]))
    frozen = json.loads(args.frozen.read_text())
    if frozen["train_identity"] != sha(json.dumps([r for r in rows if r["split"] == "train"], sort_keys=True).encode()):
        raise ValueError("baseline does not bind training bytes")
    result = evaluate(rows, frozen)
    if any(v > 0.65 for s in ("train", "validation", "audit") for k, v in result[s].items() if k != "pairs"):
        raise ValueError("shortcut gate failed")
    result["frozen_sha256"] = sha(args.frozen.read_bytes())
    write(args.out, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
