# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Nested evaluator pairs from complete originals. No neural operations."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import re

from source_gate import HERE, digest, normal, write

FAMILY = "LLM_OUTPUT_EVALUATION"
RATINGS = ("Fails", "Poor", "Acceptable", "Good", "Excellent")
FORMATS = ("json", "prose", "bare_label")
SCORES = (0, 2.5, 5, 7.5, 10)
TOKEN = re.compile(r"[a-z0-9']+")


def render(prompt, answer):
    return f"### Task:\n{prompt}\n\n### Answer:\n{answer}"


def prompt(task, output):
    # This is the proposed schema-shaped representation, not a proven live caller.
    return json.dumps({"task": task, "output": output}, ensure_ascii=False, separators=(",", ":"))


def evaluation(output, label, pretty=False):
    return json.dumps({"rating": RATINGS[label], "score": SCORES[label],
                       "assessment": "Overall helpfulness of this complete output: " + output},
                      ensure_ascii=False, indent=2 if pretty else None,
                      separators=None if pretty else (",", ":"))


def pair(original):
    label = original["helpfulness"]
    if type(label) is not int or not 0 <= label <= 4:
        raise ValueError("helpfulness annotation must be an original integer 0..4")
    if any(not isinstance(original[k], str) or not original[k].strip()
           or len(original[k].encode()) > 256 for k in ("prompt", "response")):
        raise ValueError("source task/output must be intact nonempty UTF-8 within 256 bytes")
    # Fixed two-step ordinal swap, not an invented factual critique.
    wrong = (label + 2) % 5
    # Natural JSON serializations alternate by source identity, not by label or
    # measured length. Both sides use the same schema and complete output.
    pretty = int(original["group_id"][-8:], 16) % 2 == 0
    return {"prompt": prompt(original["prompt"], original["response"]),
            "chosen": evaluation(original["response"], label, pretty),
            "rejected": evaluation(original["response"], wrong, not pretty),
            "chosen_format": "json_pretty" if pretty else "json_compact",
            "label": RATINGS[label], "wrong_label": RATINGS[wrong],
            "group_id": original["group_id"], "source_file": original["source_file"],
            "source_row": original["source_row"], "label_kind": "preference_pair",
            "annotation_kind": "constructed-rating-proxy-from-human-attribute",
            "upstream_license": "CC-BY-4.0", "human_review_of_conversion": False,
            "source_helpfulness": label, "margin": 0.0}


def connected_groups(originals):
    """Keep shared normalized prompts and outputs in one connected component."""
    parent = list(range(len(originals)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(originals):
        for field in ("prompt", "response"):
            key = normal(row[field])
            if key in seen:
                a, b = root(i), root(seen[key])
                parent[max(a, b)] = min(a, b)
            else:
                seen[key] = i
    components = {}
    for i, row in enumerate(originals):
        components.setdefault(root(i), []).append(row)
    return [(digest("\n".join(sorted({normal(r["prompt"]) for r in members})).encode()), members)
            for members in components.values()]


def tokens(text):
    return TOKEN.findall(text.casefold())


def fit_count_baseline(train):
    """Immutable integer train-fitted coefficients. No floating-point refitting."""
    coefficients = Counter()
    for row in train:
        coefficients.update(tokens(row["chosen"]))
        coefficients.subtract(tokens(row["rejected"]))
    return {"method": "train-only signed unigram counts with exact rational length normalization",
            "coefficients": dict(sorted((k, v) for k, v in coefficients.items() if v)),
            "fit_split": "train", "rows": len(train)}


def control_scores(rows, baseline):
    if not rows:
        raise ValueError("empty control split")
    metrics = Counter()
    for row in rows:
        c, r = row["chosen"], row["rejected"]
        c_tokens, r_tokens = tokens(c), tokens(r)
        def count_score(ts):
            return Fraction(sum(baseline["coefficients"].get(t, 0) for t in ts), max(1, len(ts)))
        p = json.loads(row["prompt"])
        context = set(tokens(p["task"] + " " + p["output"]))
        values = {"longer": (len(c), len(r)), "shorter": (-len(c), -len(r)),
                  "answer_only_counts": (count_score(c_tokens), count_score(r_tokens)),
                  "prefer_higher_rating": (RATINGS.index(row["label"]), RATINGS.index(row["wrong_label"])),
                  "token_overlap": (len(context & set(c_tokens)), len(context & set(r_tokens)))}
        for name, (a, b) in values.items():
            metrics[name] += 2 if a > b else 1 if a == b else 0
    return {k: v / (2 * len(rows)) for k, v in sorted(metrics.items())}


def format_pair(row, name):
    if name not in FORMATS:
        raise ValueError("unsupported claimed format")
    value = dict(row)
    p = json.loads(row["prompt"])
    for side, label_key in (("chosen", "label"), ("rejected", "wrong_label")):
        label = row[label_key]
        if name == "bare_label":
            value[side] = label
        elif name == "prose":
            value[side] = f"Overall rating: {label}.\nScore: {SCORES[RATINGS.index(label)]} out of 10.\nEvaluated output: {p['output']}"
    value["format"] = name
    return value


def validate_splits(splits):
    seen_groups, seen_text = set(), set()
    for name in ("train", "validation", "audit"):
        rows = splits[name]
        groups, texts = set(), set()
        for row in rows:
            if row["chosen"] == row["rejected"]:
                raise ValueError("identical evaluation pair")
            p = json.loads(row["prompt"])
            if set(p) != {"task", "output"}:
                raise ValueError("nested task/output contract")
            groups.add(row["group_id"])
            texts.update(normal(p[k]) for k in p)
        if groups & seen_groups or texts & seen_text:
            raise ValueError("normalized cross-split source/entity overlap")
        seen_groups |= groups
        seen_text |= texts
        if len(groups) != len(rows):
            raise ValueError("pilot requires one original per independent component")


def prepare(source, destination):
    originals = [json.loads(line) for line in (source / "intact.jsonl").read_text().splitlines()]
    candidates = {"train": [], "validation": [], "audit": []}
    excluded = Counter()
    for gid, members in sorted(connected_groups(originals)):
        official_audit = any(r["source_file"] == "validation.jsonl.gz" for r in members)
        allowed = [r for r in members if r["source_file"] == ("validation.jsonl.gz" if official_audit else "train.jsonl.gz")
                   and r["prompt"].strip() and r["response"].strip()]
        if not allowed:
            excluded["empty"] += 1
            continue
        # Pick without comparing labels, correctness or length. Preserve whole source component.
        row = min(allowed, key=lambda r: digest((r["prompt"] + "\0" + r["response"]).encode()))
        row = {**row, "group_id": gid}
        split = "audit" if official_audit else "validation" if int(gid[:8], 16) % 5 == 0 else "train"
        candidates[split].append(pair(row))
        excluded["associated_originals_not_separate_examples"] += len(members) - 1
    # Balance fit only. Never choose an audit original by model or shortcut results.
    by_label = {label: [r for r in candidates["train"] if r["label"] == label] for label in RATINGS}
    each = min(32, *(len(v) for v in by_label.values()))
    each -= each % 16
    train = sorted([r for label in RATINGS for r in by_label[label][:each]], key=lambda r: r["group_id"])
    splits = {"train": train, "validation": candidates["validation"], "audit": candidates["audit"]}
    # Extend the still-unscored provider audit with unused source-train components.
    # This allocation is fixed before any neural score. Former fit/development
    # and all associated source identities are excluded. No exposed audit reused.
    used = {r["group_id"] for rows in splits.values() for r in rows}
    reserve = [r for r in candidates["train"] if r["group_id"] not in used]
    needed = 64 - len(splits["audit"])
    extension = []
    for label in RATINGS:
        extension.extend([r for r in reserve if r["label"] == label][:needed // 5])
    selected = {r["group_id"] for r in extension}
    extension.extend([r for r in reserve if r["group_id"] not in selected][:needed - len(extension)])
    splits["audit"] = sorted(splits["audit"] + extension, key=lambda r: r["group_id"])
    validate_splits(splits)
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in splits.items():
        raw = "".join(json.dumps(r, ensure_ascii=True, sort_keys=True) + "\n" for r in rows).encode()
        (destination / (name + ".jsonl")).write_bytes(raw)
        files[name] = {"sha256": digest(raw), "rows": len(rows), "independent_components": len(rows)}
    baseline = fit_count_baseline(train)
    write(destination / "frozen-baseline.json", baseline)
    controls = control_scores(splits["validation"], baseline)
    count = Counter(r["rejected"] for r in train)
    structural = {"unique_rejected_share": len(count) / len(train) if train else 0,
                  "max_single_rejected_share": max(count.values(), default=0) / len(train) if train else 1}
    failures = []
    if len(train) < 80 or len(splits["validation"]) < 16 or len(splits["audit"]) < 64:
        failures.append("too few independent originals for complete 64-pair audit")
    if {r["label"] for r in splits["audit"]} != set(RATINGS):
        failures.append("independent audit is missing a claimed rating label")
    if structural["unique_rejected_share"] < 0.8 or structural["max_single_rejected_share"] > 0.02:
        failures.append("reused negative evaluations violate primary source limits")
    if any(v > 0.65 for v in controls.values()):
        failures.append("validation shortcut exceeds 65 percent")
    report = {"family": FAMILY, "files": files, "structural": structural,
              "validation_controls": controls, "training_allowed": not failures, "failures": failures,
              "candidate_counts": {k: len(v) for k, v in candidates.items()}, "exclusions": dict(excluded),
              "fit_labels": {k: len([r for r in train if r["label"] == k]) for k in RATINGS},
              "audit_controls_measured": False, "source_inventory_sha256": digest((source / "source-inventory.json").read_bytes()),
              "scope": "constructed evaluator helpfulness rating proxy, not universal factual proof",
              "source_and_output_truncation": False, "qualification_template": "literal primary RenderForJudgment",
              "miner_application_body_selection": "not claimed",
              "audit_extension_policy": "pre-model source-stratified complementary components, no exposed audits",
              "claimed_formats": list(FORMATS)}
    write(destination / "data-report.json", report)
    print(json.dumps(report, sort_keys=True))
    return report


def load_splits(root):
    report = json.loads((root / "data-report.json").read_text())
    splits = {}
    for name, value in report["files"].items():
        raw = (root / (name + ".jsonl")).read_bytes()
        if digest(raw) != value["sha256"]:
            raise ValueError("frozen split hash mismatch")
        splits[name] = [json.loads(line) for line in raw.splitlines()]
        if len(splits[name]) != value["rows"]:
            raise ValueError("frozen split count mismatch")
    validate_splits(splits)
    return report, splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=HERE / "evidence/source")
    parser.add_argument("--out", type=Path, default=HERE / "evidence/data")
    args = parser.parse_args()
    prepare(args.source, args.out)
