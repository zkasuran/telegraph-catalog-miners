# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Data identity checks and immutable train-fitted integer baselines."""
import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from scorer import digest, render, score, strict_json

WORDS = re.compile(r"[a-z0-9']+")
SPLITS = ("train", "validation", "audit")


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def load(root):
    return {s: [strict_json(line) for line in (root/(s+".jsonl")).read_text().split("\n") if line]
            for s in SPLITS}


def fit(rows):
    # Integer signed token frequencies, no platform-dependent floating fit.
    coefficients = Counter()
    for row in rows:
        coefficients.update(WORDS.findall(row["chosen"].casefold()))
        coefficients.subtract(WORDS.findall(row["rejected"].casefold()))
    differences = [len(r["chosen"])-len(r["rejected"]) for r in rows]
    direction = 1 if sum(d > 0 for d in differences) >= sum(d < 0 for d in differences) else -1
    return {"schema_version": 1, "fit_split": "train", "length_direction": direction,
            "answer_only_integer_coefficients": {w: c for w, c in sorted(coefficients.items()) if c}}


def half_accuracy(values):
    return {"correct_twice": sum(1 if d == 0 else 2 if d > 0 else 0 for d in values), "pairs": len(values)}


def checks(splits, baseline):
    groups, answers, prompt_ids, reports = {}, {}, {}, {}
    summaries = {}
    for split, rows in splits.items():
        if not rows:
            raise ValueError("empty split")
        seen, source_ids, states = set(), set(), Counter()
        for row in rows:
            if row["split"] != split:
                raise ValueError("split field mismatch")
            group, package = row["group_id"], row["source_id"]
            match = re.fullmatch(r"BILLS-(113|115|117|119)(?:hr|s|hjres|sjres|hconres|sconres)\d+enr", package)
            if not match or {"113": "train", "115": "train", "117": "validation", "119": "audit"}[match[1]] != split:
                raise ValueError("publication-session split mismatch")
            if group in groups and groups[group] != split:
                raise ValueError("overlapping original digest")
            if package in reports and reports[package] != split:
                raise ValueError("overlapping bill/revision family")
            groups[group], reports[package] = split, split
            task = strict_json(row["prompt"])
            if task["report"]["package"] != package:
                raise ValueError("source identity mismatch")
            if row["variant"] == "original" and task["report"]["sha256"] != group:
                raise ValueError("original digest mismatch")
            source_ids.add(package)
            identity = (row["group_id"], row["variant"], row["field"])
            if identity in seen:
                raise ValueError("duplicate pair view")
            seen.add(identity)
            # A deterministic authority is an oracle baseline, not a learned win.
            if score(row["prompt"], row["chosen"])["score"] != 1 or score(row["prompt"], row["rejected"])["score"] != 0:
                raise ValueError("constructed preference contradicts measured report")
            states[task["report"][row["field"]]] += 1
            key = digest(normalize(row["prompt"]).encode())
            if key in prompt_ids and prompt_ids[key] != split:
                raise ValueError("normalized cross-split prompt overlap")
            prompt_ids[key] = split
            for side in ("chosen", "rejected"):
                key = digest(normalize(row[side]).encode())
                if key in answers and answers[key] != split:
                    raise ValueError("normalized cross-split answer overlap")
                answers[key] = split
        if states[True] != states[False]:
            raise ValueError("measured assertion imbalance")
        weights = baseline["answer_only_integer_coefficients"]
        def bow(text):
            return sum(weights.get(w, 0) for w in WORDS.findall(text.casefold()))
        def overlap(prompt, answer):
            return len(set(WORDS.findall(prompt.casefold())) & set(WORDS.findall(answer.casefold())))
        shortcuts = {
            "length": half_accuracy([(len(r["chosen"])-len(r["rejected"]))*baseline["length_direction"] for r in rows]),
            "answer_only": half_accuracy([bow(r["chosen"])-bow(r["rejected"]) for r in rows]),
            "token_overlap": half_accuracy([overlap(r["prompt"], r["chosen"])-overlap(r["prompt"], r["rejected"]) for r in rows])}
        if any(v["correct_twice"]*100 > v["pairs"]*130 for v in shortcuts.values()):
            raise ValueError("shortcut exceeds unchanged 65 percent gate: " + json.dumps({"split": split, "baselines": shortcuts}))
        summaries[split] = {"pairs": len(rows), "originals": len(source_ids), "assertions_true": states[True],
                            "assertions_false": states[False], "shortcut_baselines": shortcuts,
                            "max_render_utf8_bytes": max(len(render(r["prompt"], r[a]).encode()) for r in rows for a in ("chosen", "rejected")),
                            "oracle_scope": "constructed report-consistency, not external authenticity"}
    return {"splits": summaries, "normalized_cross_split_overlap": 0, "original_groups": len(groups),
            "audit": "data invariants only; not candidate selection or neural audit"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    splits = load(args.data)
    path = args.data/"baseline.json"
    if args.freeze:
        if path.exists():
            raise ValueError("baseline is already frozen; never refit it")
        path.write_text(json.dumps(fit(splits["train"]), indent=2, sort_keys=True)+"\n")
    baseline = json.loads(path.read_text())
    try:
        report = checks(splits, baseline)
    except ValueError as exc:
        (args.data/"data-failure.json").write_text(json.dumps({"failure": str(exc), "baseline_sha256": digest(path.read_bytes())}, indent=2)+"\n")
        raise
    report["files"] = {s: digest((args.data/(s+".jsonl")).read_bytes()) for s in SPLITS}
    report["baseline_sha256"] = digest(path.read_bytes())
    (args.data/"data-check.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
