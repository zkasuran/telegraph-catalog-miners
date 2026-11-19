# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Original counterbalanced DNS-question fixtures. No observed DNS labels."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from scorer import inspect, parse_json, request_prompt, render

HERE = Path(__file__).resolve().parent
SIZES = {"train": 128, "validation": 32, "audit": 64}  # Groups, two paired requests each.


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2) + "\n").encode()


def write_json(path, value):
    path.write_bytes(encoded(value))


def words(text):
    return re.findall(r"[a-z0-9_]+", text.lower())


def audit_labels(row):
    body = parse_json(row["chosen"])
    return (row["control"], "lookup_ok_true" if body["Status"] == 0 else "lookup_ok_false")


def make_groups(split):
    rows = []
    for i in range(SIZES[split]):
        tag = digest(f"email-security-question-binding-v1:{split}:{i}".encode())[:16]
        group = "zone-" + tag
        # These are authored .test zones, not invented observations or real
        # names scraped from a protected evaluation question.
        zone = "z" + tag + ".test"
        base = ("_dmarc." if i % 4 == 0 else "mail.") + zone
        name_a, name_b = base, "alt." + zone
        type_a, type_b = (("TXT", "MX") if i % 2 else ("MX", "TXT"))
        mode = "name" if i % 2 else "type"
        if mode == "name":
            type_b = type_a
        else:
            name_b = name_a
        code = (0, 2, 3, 5)[(i // 2) % 4]
        bodies = []
        for name, kind in ((name_a, type_a), (name_b, type_b)):
            body = {"Status": code, "Question": [{"name": name + ".", "type": 16 if kind == "TXT" else 15}]}
            # Do not add a correctness flag or annotation to the answer text.
            # Audit uses a serialization layout absent from train/validation.
            if split == "train":
                answer = json.dumps(body, separators=(",", ":"))
            elif split == "validation":
                answer = json.dumps({"Question": body["Question"], "Status": code}, separators=(", ", ": "))
            else:
                answer = json.dumps(body, indent=1)
            bodies.append(answer)
        for side, (name, kind) in enumerate(((name_a, type_a), (name_b, type_b))):
            rows.append({"group_id": group, "zone": zone, "pair_id": group + ":" + str(side),
                         "prompt": request_prompt(name.upper() if i % 8 == 0 else name, kind),
                         "chosen": bodies[side], "rejected": bodies[1 - side], "margin": 0.0,
                         "label_kind": "preference_pair", "control": mode,
                         "authorship": "AI-assisted original constructed protocol fixture",
                         "source_observation": False})
    return rows


def check_groups(splits):
    seen_groups, seen_prompts, seen_queries, seen_answers = set(), set(), set(), set()
    for split, rows in splits.items():
        if len(rows) != 2 * SIZES[split]:
            raise ValueError("fixed row count")
        grouped = {}
        current_prompts, current_queries, current_answers = set(), set(), set()
        for row in rows:
            if row["source_observation"] is not False:
                raise ValueError("constructed fixture was relabelled as observed")
            grouped.setdefault(row["group_id"], []).append(row)
            prompt = " ".join(row["prompt"].lower().split())
            current_prompts.add(prompt)
            for side in ("chosen", "rejected"):
                value = parse_json(row[side])
                query = value["Question"][0]
                current_queries.add((query["name"].lower().rstrip("."), query["type"]))
                current_answers.add(json.dumps(value, sort_keys=True, separators=(",", ":")).lower())
                render(row["prompt"], row[side])
            if inspect(row["prompt"], row["chosen"])["score"] != 1 or inspect(row["prompt"], row["rejected"])["score"] != -1:
                raise ValueError("fixture fails exact question binding")
        for group, pair in grouped.items():
            if len(pair) != 2 or pair[0]["chosen"] != pair[1]["rejected"] or pair[1]["chosen"] != pair[0]["rejected"]:
                raise ValueError("answer-only counterbalance missing")
        if (seen_groups & grouped.keys() or seen_prompts & current_prompts
                or seen_queries & current_queries or seen_answers & current_answers):
            raise ValueError("normalized split overlap")
        seen_groups.update(grouped)
        seen_prompts.update(current_prompts)
        seen_queries.update(current_queries)
        seen_answers.update(current_answers)
    return {"groups": len(seen_groups), "normalized_split_overlaps": 0,
            "group_policy": "whole authored zone and reversed answer/query siblings together",
            "source_held_out": False, "time_held_out": False,
            "audit_layout_held_out": True, "original_observed_dns_records": 0}


def freeze_controls(train):
    # Signed integer frequency coefficients fit once on train. Counterbalancing
    # makes every coefficient exactly zero, without floating-point fitting.
    coeff = Counter()
    for row in train:
        coeff.update(words(row["chosen"]))
        coeff.subtract(words(row["rejected"]))
    return {"SPDX-License-Identifier": "LicenseRef-zkasuran-SAND-1.0",
            "schema_version": 1, "fit_split": "train", "algorithm": "signed-token-frequency-v1",
            "coefficients": dict(sorted(coeff.items())), "length_orientation": 1,
            "fit_rows": len(train), "floating_point_fit": False}


def evaluate_controls(rows, frozen):
    def score(text):
        return sum(frozen["coefficients"].get(word, 0) for word in words(text))
    def accuracy(differences):
        return sum(1 if x > 0 else .5 if x == 0 else 0 for x in differences) / len(rows)
    overlap = []
    for row in rows:
        prompt = set(words(row["prompt"]))
        overlap.append(len(prompt & set(words(row["chosen"]))) - len(prompt & set(words(row["rejected"]))))
    return {"length": accuracy([len(r["chosen"]) - len(r["rejected"]) for r in rows]),
            "answer_only": accuracy([score(r["chosen"]) - score(r["rejected"]) for r in rows]),
            "literal_word_overlap": accuracy(overlap),
            "exact_question_parser": sum(inspect(r["prompt"], r["chosen"])["score"] >
                                         inspect(r["prompt"], r["rejected"])["score"] for r in rows) / len(rows)}


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    splits = {name: make_groups(name) for name in SIZES}
    report = check_groups(splits)
    files = {}
    for name, rows in splits.items():
        raw = b"".join(json.dumps(r, sort_keys=True, ensure_ascii=True, allow_nan=False).encode() + b"\n" for r in rows)
        path = root / (name + ".jsonl")
        if path.exists() and path.read_bytes() != raw:
            raise ValueError("refusing to replace existing split bytes")
        path.write_bytes(raw)
        files[name] = {"sha256": digest(raw), "rows": len(rows), "groups": SIZES[name]}
    frozen = freeze_controls(splits["train"])
    write_json(root / "frozen-controls.json", frozen)
    report.update({"files": files, "family": "EMAIL_SECURITY", "scope": "DNS Question identity binding only",
                   "freeze_sha256": digest(encoded(frozen)), "audit_metrics": "not evaluated before neural candidate freeze",
                   "validation_controls": evaluate_controls(splits["validation"], frozen),
                   "licence": "LicenseRef-zkasuran-SAND-1.0", "human_reviewed": False})
    write_json(root / "data-report.json", report)
    return report


def read_splits(root):
    report = json.loads((root / "data-report.json").read_bytes())
    splits = {}
    for name, size in SIZES.items():
        raw = (root / (name + ".jsonl")).read_bytes()
        if len(raw) > 4 << 20 or digest(raw) != report["files"][name]["sha256"]:
            raise ValueError("split hash or byte ceiling")
        splits[name] = [json.loads(line) for line in raw.splitlines()]
        if len(splits[name]) != 2 * size:
            raise ValueError("split count changed")
    check_groups(splits)
    frozen_raw = (root / "frozen-controls.json").read_bytes()
    if digest(frozen_raw) != report["freeze_sha256"]:
        raise ValueError("frozen baseline hash mismatch")
    return report, splits, json.loads(frozen_raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    print(json.dumps(prepare(parser.parse_args().out), indent=2))
