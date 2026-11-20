# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Replay the original full UCI inventory and grouped splits, without models."""
import argparse
import json
import math
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from contracts import FAMILY, FROZEN, LABELS, fetch, sha, strict_json, write_json

WORDS = re.compile(r"[a-z0-9']+")


def group_id(text):
    return sha(" ".join(unicodedata.normalize("NFKC", text).casefold().split()).encode())


def inventory(archive):
    rows, counts = [], {}
    with zipfile.ZipFile(archive) as source:
        for domain, filename in (("amazon", "amazon_cells"), ("imdb", "imdb"), ("yelp", "yelp")):
            member = "sentiment labelled sentences/" + filename + "_labelled.txt"
            if source.getinfo(member).file_size > 1 << 20:
                raise ValueError("oversized source inventory member")
            total = 0
            for number, line in enumerate(source.read(member).decode().split("\n"), 1):
                if not line.strip():
                    continue
                text, label = line.rstrip("\r").rsplit("\t", 1)
                if label not in ("0", "1"):
                    raise ValueError("invalid original source annotation")
                group = group_id(text)
                split = "test" if domain == "yelp" else "validation" if int(group[:8], 16) % 8 == 0 else "train"
                rows.append({"text": text, "label": "Positive" if label == "1" else "Negative",
                             "split": split, "domain": domain, "source_id": f"{domain}:{number}"})
                total += 1
            counts[domain] = total
    if counts != {"amazon": 1000, "imdb": 1000, "yelp": 1000}:
        raise ValueError("full original source inventory count mismatch")
    return rows, counts


def select(rows):
    groups = defaultdict(list)
    for row in rows:
        text = row["text"].strip()
        if 8 <= len(text.encode()) <= 256:
            groups[group_id(text)].append({**row, "text": text})
    buckets, removed = defaultdict(list), 0
    for group, occurrences in groups.items():
        if len({r["split"] for r in occurrences}) != 1 or len({r["label"] for r in occurrences}) != 1:
            removed += len(occurrences)
            continue
        row = occurrences[0]
        buckets[row["split"], row["label"]].append({**row, "group_id": group})
    splits = {}
    for split, spec in FROZEN["splits"].items():
        chosen = []
        for label in sorted(LABELS):
            candidates = sorted(buckets[split, label], key=lambda r: sha(("select-v1:" + r["group_id"]).encode()))
            if len(candidates) < spec["pairs"] // 2:
                raise ValueError("insufficient original source groups")
            chosen.extend(candidates[:spec["pairs"] // 2])
        splits[split] = sorted(chosen, key=lambda r: r["group_id"])
    return splits, removed


def pair(row, style=None):
    text, label = row["text"], row["label"]
    opposite = {"Positive": "Negative", "Negative": "Positive"}[label]
    style = int(row["group_id"][:2], 16) % 2 if style is None else style
    if style not in (0, 1, 2):
        raise ValueError("unsupported response format")

    def answer(value):
        if style == 2:
            return value
        if style == 1:
            return f"Sentiment: {value}. Text assessed: {text}"
        return json.dumps({"sentiment": value, "reason": "Text assessed: " + text}, ensure_ascii=False, separators=(",", ":"))

    return {"prompt": json.dumps({"text": text}, ensure_ascii=False, separators=(",", ":")),
            "chosen": answer(label), "rejected": answer(opposite), "label_kind": "preference_pair",
            "label": label, "error_family": "label_flip", "group_id": row["group_id"],
            "source_id": row["source_id"], "domain": row["domain"], "style": style}


def read_rows(path):
    return [strict_json(line) for line in path.read_bytes().decode().split("\n") if line.strip()]


def check_splits(splits):
    seen = set()
    for name, rows in splits.items():
        if len(rows) != FROZEN["splits"][name]["pairs"]:
            raise ValueError("frozen source split count mismatch")
        identifiers = []
        for row in rows:
            text = strict_json(row["prompt"])["text"]
            if row["group_id"] != group_id(text) or not 8 <= len(text.encode()) <= 256:
                raise ValueError("source group or length mismatch")
            if (name == "test") != (row["domain"] == "yelp"):
                raise ValueError("source domain crossed audit boundary")
            identifiers.append(row["group_id"])
        if len(set(identifiers)) != len(rows) or seen.intersection(identifiers):
            raise ValueError("normalized source overlap")
        seen.update(identifiers)
        if Counter(row["label"] for row in rows) != {label: len(rows) // 2 for label in LABELS}:
            raise ValueError("source labels not balanced")


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "sentiment.zip"
    fetch(FROZEN["source_url"], archive, FROZEN["source_sha256"], 1 << 20)
    originals, counts = inventory(archive)
    selected, removed = select(originals)
    splits = {name: [pair(row) for row in rows] for name, rows in selected.items()}
    check_splits(splits)
    files = {}
    for name, rows in splits.items():
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        if sha(raw) != FROZEN["splits"][name]["sha256"]:
            raise ValueError("exact original source split hash mismatch")
        (root / (name + ".jsonl")).write_bytes(raw)
        files[name] = {"sha256": sha(raw), "pairs": len(rows), "labels": dict(Counter(r["label"] for r in rows))}
    report = {"family": FAMILY, "archive_sha256": FROZEN["source_sha256"], "full_inventory": counts,
              "files": files, "normalized_overlap": 0, "removed_conflicting_occurrences": removed,
              "independent_audit_groups": len(splits["test"]), "audit_used_for_selection": False,
              "training_identity": {"run_id": FROZEN["training_run_id"], "commit": FROZEN["training_commit"]},
              "construction": "Original CC BY source labels; AI-assisted programmatic preference pairs, not human-reviewed preferences.",
              "scope": FROZEN["scope"]}
    write_json(root / "source-manifest.json", report)
    return splits, report


def features(row, coefficient):
    idf = coefficient["idf"]

    def vector(text):
        return {word: count * idf[word] for word, count in Counter(WORDS.findall(text.casefold())).items() if word in idf}

    def cosine(a, b):
        denominator = math.sqrt(sum(v*v for v in a.values()) * sum(v*v for v in b.values()))
        return sum(value * b.get(word, 0) for word, value in a.items()) / denominator if denominator else 0.0

    prompt, chosen, rejected = vector(row["prompt"]), vector(row["chosen"]), vector(row["rejected"])
    def keywords(text):
        return WORDS.findall(text.casefold()).count("negative")
    return {"length": len(row["chosen"]) - len(row["rejected"]),
            "tfidf_overlap": cosine(prompt, chosen) - cosine(prompt, rejected),
            "alarm_keywords": keywords(row["chosen"]) - keywords(row["rejected"])}


def shortcuts(rows, coefficient):
    differences = [features(row, coefficient) for row in rows]
    result = {name: sum(.5 if abs(row[name]) < 1e-10 else float(row[name] * direction > 0) for row in differences) / len(rows)
              for name, direction in coefficient["direction"].items()}
    # A constant answer-label preference is trained on the balanced training labels.
    result["answer_only"] = sum(row["label"] == coefficient["answer_label"] for row in rows) / len(rows)
    return result


def fit_training_only(train_path, output):
    raw = train_path.read_bytes()
    if sha(raw) != FROZEN["splits"]["train"]["sha256"]:
        raise ValueError("baseline fit accepts only the frozen training split")
    train = read_rows(train_path)
    frequencies = Counter(word for row in train for word in set(WORDS.findall(row["prompt"].casefold())))
    idf = {word: math.log((1 + len(train)) / (1 + count)) + 1 for word, count in sorted(frequencies.items())}
    coefficient = {"SPDX-License-Identifier": "CC-BY-4.0", "schema_version": 1, "train_sha256": sha(raw), "idf": idf,
                   "direction": {"length": 1, "tfidf_overlap": 1, "alarm_keywords": 1}, "answer_label": "Negative",
                   "fit_scope": "Frozen public training split only; coefficients serialized once, never refitted in release verification."}
    scores = shortcuts(train, coefficient)
    coefficient["direction"] = {name: 1 if scores[name] >= .5 else -1 for name in coefficient["direction"]}
    write_json(output, coefficient)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fit-baseline", type=Path)
    args = parser.parse_args()
    _, report = prepare(args.out)
    if args.fit_baseline:
        fit_training_only(args.out / "train.jsonl", args.fit_baseline)
    print(json.dumps({"full_inventory": report["full_inventory"], "normalized_overlap": 0, "files": report["files"]}))


if __name__ == "__main__":
    main()
