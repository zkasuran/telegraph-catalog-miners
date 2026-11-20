# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-aware grouped split checks and immutable integer shortcut coefficients."""
import json
from collections import Counter

from scorer import tokens
from source import norm, sha

SPLITS = ("train", "validation", "test")


def check_pilot_counts(splits):
    minimums = {"train": 64, "validation": 16, "test": 64}
    if any(len(splits[s]) < minimums[s] for s in SPLITS):
        raise ValueError("insufficient reviewed pairs for the 64/16/64 pilot")
    if any(len({r["group_id"] for r in splits[s]}) < minimums[s] // 2 for s in SPLITS):
        raise ValueError("insufficient independent components for the pilot")
    return True


def check_splits(splits):
    if set(splits) != set(SPLITS):
        raise ValueError("all three grouped splits are required")
    seen = {k: set() for k in ("group_id", "doi", "author", "text", "answer")}
    for split in SPLITS:
        rows = splits[split]
        if not rows:
            raise ValueError("empty approved split: " + split)
        current = {k: set() for k in seen}
        for row in rows:
            if row.get("label_kind") != "preference_pair" or row.get("review_origin") not in {"AI-assisted", "human-reviewed"}:
                raise ValueError("unreviewed or unknown preference origin")
            if not row.get("centrality_reason") or not row.get("source"):
                raise ValueError("preference must retain reasoning and source attribution")
            source = row["source"]
            if source.get("licence") not in {"CC-BY-3.0", "CC-BY-4.0", "CC0-1.0"} or not source.get("grant") or not source.get("url"):
                raise ValueError("missing original source grant")
            if row["chosen"] == row["rejected"]:
                raise ValueError("identical preference responses")
            current["group_id"].add(row["group_id"])
            current["doi"].add(norm(source["doi"]))
            current["text"].add(norm(source["text"]))
            current["author"].update(norm(a) for a in source["authors"])
            current["answer"].update(norm(row[s]) for s in ("chosen", "rejected"))
            if row.get("off_topic_donor"):
                donor = row.get("donor_source")
                if split != "train" or not isinstance(donor, dict) or donor.get("licence") not in {"CC-BY-3.0", "CC-BY-4.0", "CC0-1.0"} or not donor.get("grant") or not donor.get("url"):
                    raise ValueError("precision donor requires attributed training-only source")
                current["doi"].add(norm(donor["doi"]))
                current["text"].add(norm(donor["text"]))
                current["author"].update(norm(a) for a in donor["authors"])
        for field, values in seen.items():
            if current[field] & values:
                raise ValueError("normalized cross-split " + field + " overlap")
            values.update(current[field])
        donors = sum(bool(r.get("off_topic_donor")) for r in rows)
        if donors / len(rows) > 0.25:
            raise ValueError("off-topic donor fraction exceeds 25%")
    return True


def features(row):
    prompt_words = set(tokens(row["prompt"]))
    def overlap(value):
        words = set(tokens(value))
        return len(words & prompt_words) * 1000000 // max(1, len(words))
    return {"length": len(row["chosen"]) - len(row["rejected"]),
            "overlap": overlap(row["chosen"]) - overlap(row["rejected"])}


def fit_baselines(train):
    if not train:
        raise ValueError("baseline fit requires training rows")
    def correct(diff):
        return 1 if diff == 0 else 2 if diff > 0 else 0
    values = [features(r) for r in train]
    signs = {key: 1 if sum(correct(v[key]) for v in values) >= len(values) else -1 for key in values[0]}
    words = Counter()
    for r in train:
        words.update(set(tokens(r["chosen"])))
        words.subtract(set(tokens(r["rejected"])))
    return {"schema_version": 1, "fit_split": "train", "directions": signs,
            "answer_only_integer_coefficients": dict(sorted(words.items())),
            "train_sha256": sha(json.dumps(train, ensure_ascii=False, sort_keys=True).encode()),
            "coefficient_rule": "Chosen minus rejected document frequency. Integer arithmetic only; no fresh fitting during reproduction."}


def shortcut_report(rows, frozen):
    if not rows or frozen.get("fit_split") != "train":
        raise ValueError("empty evaluation or unfrozen coefficients")
    totals = Counter()
    for row in rows:
        differences = features(row)
        for key, direction in frozen["directions"].items():
            d = differences[key] * direction
            totals[key] += 1 if d == 0 else 2 if d > 0 else 0
        weights = frozen["answer_only_integer_coefficients"]
        difference = sum(weights.get(w, 0) for w in set(tokens(row["chosen"]))) - sum(weights.get(w, 0) for w in set(tokens(row["rejected"])))
        totals["answer_only"] += 1 if difference == 0 else 2 if difference > 0 else 0
    return {key: total / (2 * len(rows)) for key, total in totals.items()}
