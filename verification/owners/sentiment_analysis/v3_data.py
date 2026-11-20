# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""New unanimous-source splits, prior-audit quarantine and short response evidence."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from contracts import FAMILY, FROZEN, HERE, fetch, file_sha, sha, strict_json
from retrain_data import LABELS, SOURCE_FILE, baselines as old_baselines, download, dump
from source_data import WORDS, group_id, inventory

COUNTS = {"train": 192, "validation": 48, "audit": 48}
BASELINE = HERE / "v3-baseline.json"
QUARANTINE = HERE / "v3-quarantine.json"
POSITIVE = {"amusement", "excitement", "joy", "love", "optimism", "pride", "admiration", "gratitude", "relief"}


def unanimous_inventory(raw_root, mapping):
    originals = {}
    totals = {}
    positive = POSITIVE
    negative = set(mapping["negative"])
    excluded_emotions = set(mapping["ambiguous"]) | (set(mapping["positive"]) - positive)
    emotion_names = positive | negative | excluded_emotions | {"neutral"}
    for part in range(1, 4):
        name = f"goemotions_{part}.csv"
        count = 0
        with (raw_root / name).open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not emotion_names | {"id", "text", "rater_id", "created_utc", "author", "link_id", "example_very_unclear"} <= set(reader.fieldnames or []):
                raise ValueError("full annotation schema mismatch")
            for row in reader:
                count += 1
                if count > 100000 or any(row[e] not in ("0", "1") for e in emotion_names):
                    raise ValueError("annotation value/count ceiling")
                timestamp = float(row["created_utc"])
                if not math.isfinite(timestamp):
                    raise ValueError("nonfinite source timestamp")
                current = {"source_id": row["id"], "text": row["text"].strip(),
                           "author": sha(row["author"].encode()), "thread": sha(row["link_id"].encode()),
                           "timestamp": int(timestamp), "anonymous": row["author"] in ("", "[deleted]")}
                original = originals.setdefault(row["id"], {**current, "votes": {}, "conflict": False})
                if any(original[key] != value for key, value in current.items()):
                    original["conflict"] = True
                pos = any(row[e] == "1" for e in positive)
                neg = any(row[e] == "1" for e in negative)
                excluded = any(row[e] == "1" for e in excluded_emotions)
                neutral = row["neutral"] == "1"
                vote = ("Positive" if pos and not neg and not neutral and not excluded else
                        "Negative" if neg and not pos and not neutral and not excluded else
                        "Neutral" if neutral and not pos and not neg and not excluded else None)
                if row["example_very_unclear"].casefold() in ("true", "1"):
                    vote = None
                if row["rater_id"] in original["votes"] and original["votes"][row["rater_id"]] != vote:
                    original["conflict"] = True
                original["votes"][row["rater_id"]] = vote
        totals[name] = count
    if totals != {"goemotions_1.csv": 70000, "goemotions_2.csv": 70000, "goemotions_3.csv": 71225}:
        raise ValueError("complete source inventory required")
    times = sorted(row["timestamp"] for row in originals.values())
    cutoffs = (times[len(times)*3//4], times[len(times)*7//8])
    rows = []
    excluded = Counter()
    for original in originals.values():
        values = list(original["votes"].values())
        if (original["conflict"] or original["anonymous"] or not 8 <= len(original["text"].encode()) <= 128 or
                len(values) < 3 or len(set(values)) != 1 or values[0] is None):
            excluded["not_unanimous_three_raters_or_bounded_source"] += 1
            continue
        row = {key: original[key] for key in ("source_id", "text", "author", "thread", "timestamp")}
        row.update(label=values[0], group_id=group_id(row["text"]), rater_agreement=len(values),
                   split="train" if row["timestamp"] < cutoffs[0] else "validation" if row["timestamp"] < cutoffs[1] else "audit")
        rows.append(row)
    return rows, {"raw_rows": totals, "distinct_comments": len(originals), "time_cutoffs": cutoffs,
                  "eligible_unanimous_comments": len(rows), "excluded": dict(excluded)}


def select_groups(rows, prior_groups):
    quarantine = strict_json(QUARANTINE.read_bytes())
    forbidden = {key: set(quarantine[key]) for key in ("sources", "entities", "threads", "source_groups")}
    memberships = defaultdict(set)
    for row in rows:
        for key in ("author", "thread", "group_id"):
            memberships[key, row[key]].add(row["split"])
    eligible = defaultdict(list)
    excluded = Counter()
    for row in rows:
        if (row["source_id"] in forbidden["sources"] or row["author"] in forbidden["entities"] or
                row["thread"] in forbidden["threads"] or row["group_id"] in forbidden["source_groups"] or
                row["group_id"] in prior_groups):
            excluded["prior_audit_or_uci_identity"] += 1
        elif any(len(memberships[key, row[key]]) != 1 for key in ("author", "thread", "group_id")):
            excluded["cross_time_entity_or_text"] += 1
        else:
            eligible[row["split"], row["label"]].append(row)
    used = set()
    splits = {}
    for split, n in COUNTS.items():
        pool = {}
        for label in LABELS[:3]:
            need = n + n//3 + (n if label != "Neutral" else 0)
            pool[label] = []
            for row in sorted(eligible[split, label], key=lambda r: sha(("sentiment-v3:" + r["group_id"]).encode())):
                keys = {(key, row[key]) for key in ("author", "thread", "group_id")}
                if used & keys:
                    continue
                used.update(keys)
                pool[label].append(row)
                if len(pool[label]) == need:
                    break
            if len(pool[label]) != need:
                raise ValueError(f"insufficient new independent {split}/{label}: {len(pool[label])}/{need}")
        selected = []
        for label in LABELS[:3]:
            simple = n*2//3
            for row in pool[label][:simple]:
                selected.append(make_group([row], label))
            for index in range(n//3):
                offset = simple + 2*index
                selected.append(make_group(pool[label][offset:offset+2], label, index))
        offset = n + n//3
        for index, (positive, negative) in enumerate(zip(pool["Positive"][offset:], pool["Negative"][offset:])):
            selected.append(make_group([positive, negative] if index % 2 == 0 else [negative, positive], "Mixed", index))
        splits[split] = selected
    check_groups(splits, quarantine)
    return splits, {"eligible": {f"{split}/{label}": len(values) for (split, label), values in sorted(eligible.items())},
                    "excluded": dict(excluded), "quarantine_sha256": file_sha(QUARANTINE)}


def make_group(parts, label, index=0):
    connector = (" ", " Also, ", " However, ", " Yet, ", " and ", " but ")[index % 6]
    text = connector.join(row["text"] for row in parts)
    return {"text": text, "label": label, "group_id": group_id(text),
            "sources": [row["source_id"] for row in parts], "entities": [row["author"] for row in parts],
            "threads": [row["thread"] for row in parts], "source_groups": [row["group_id"] for row in parts],
            "source_times": [row["timestamp"] for row in parts], "composition": len(parts),
            "construction": "unanimous upstream emotion-to-valence mapping" if len(parts) == 1 else
                            "constructed composition of independent upstream comments; not human-reviewed sentiment"}


def check_groups(splits, quarantine):
    used = set()
    for split, rows in splits.items():
        if Counter(row["label"] for row in rows) != {label: COUNTS[split] for label in LABELS}:
            raise ValueError("new split must retain all four balanced labels")
        for row in rows:
            if row["group_id"] != group_id(row["text"]) or row["group_id"] in quarantine["constructed_groups"]:
                raise ValueError("constructed source identity or quarantine mismatch")
            keys = {(key, value) for key in ("sources", "entities", "threads", "source_groups") for value in row[key]}
            if (len(keys) != sum(len(row[key]) for key in ("sources", "entities", "threads", "source_groups")) or
                    used & keys or any(set(row[key]) & set(quarantine[key]) for key in ("sources", "entities", "threads", "source_groups"))):
                raise ValueError("new source/entity/thread overlap or prior audit reuse")
            used.update(keys)


def pairs(rows, style=None, training=False):
    result = []
    for label in LABELS:
        for index, row in enumerate(sorted((r for r in rows if r["label"] == label), key=lambda r: r["group_id"])):
            text = row["text"]
            prompt = (json.dumps({"text": text}, ensure_ascii=False, separators=(",", ":")) if index % 4 == 0 else
                      text if index % 4 == 1 else "Classify the overall sentiment of this text: " + text if index % 4 == 2 else
                      "Choose Positive, Negative, Neutral or Mixed for the following text.\n" + text)
            selected = (2 if training and index % 16 == 0 else index % 2) if style is None else style
            if selected not in (0, 1, 2):
                raise ValueError("unsupported response style")
            words = text.split()
            excerpt = " ".join(words[:4] + (["..."] + words[-4:] if len(words) > 8 else words[4:]))

            def answer(value):
                if selected == 2:
                    return value
                if selected == 1:
                    return f"Sentiment: {value}. Text excerpt: {excerpt}"
                return json.dumps({"sentiment": value, "score": {"Positive": 1, "Negative": -1, "Neutral": 0, "Mixed": 0}[value],
                                   "reason": "Text excerpt: " + excerpt}, ensure_ascii=False, separators=(",", ":"))
            for wrong in LABELS:
                if wrong != label:
                    result.append({"prompt": prompt, "chosen": answer(label), "rejected": answer(wrong),
                                   "label": label, "wrong_label": wrong, "label_kind": "preference_pair",
                                   "group_id": row["group_id"], "composition": row["composition"], "margin": 0.0})
    return result


def prompt_bin(row, kind):
    if kind == "prompt_length":
        return str(min(len(row["prompt"].encode()) // 48, 8))
    return str(any(token in row["prompt"].casefold() for token in ("however", " but ", " yet ", " also, ")))


def baselines(rows, coefficient):
    result = old_baselines(rows, coefficient)
    for kind, buckets in coefficient["prompt_histograms"].items():
        hits = 0
        for row in rows:
            bucket = buckets.get(prompt_bin(row, kind), {})
            chosen, rejected = bucket.get(row["label"], 0), bucket.get(row["wrong_label"], 0)
            hits += .5 if chosen == rejected else float(chosen > rejected)
        result[kind] = hits / len(rows)
    return result


def prepare(root, source_root=None, freeze=False):
    root.mkdir(parents=True, exist_ok=True)
    raw_root = source_root or root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    source = strict_json(SOURCE_FILE.read_bytes())
    if source["stage"] != "verified":
        raise ValueError("reviewed source grants/hashes required")
    for name, spec in source["files"].items():
        download(spec["url"], raw_root / name, spec["sha256"])
    rows, inventory_report = unanimous_inventory(raw_root, strict_json((raw_root / "sentiment_mapping.json").read_bytes()))
    prior = root / "excluded-uci.zip"
    fetch(FROZEN["source_url"], prior, FROZEN["source_sha256"], 1 << 20)
    uci, _ = inventory(prior)
    groups, selection = select_groups(rows, {group_id(row["text"]) for row in uci})
    files = {}
    training = None
    for split, originals in groups.items():
        dump(root / (split + "-groups.json"), originals)
        built = pairs(originals, training=split == "train")
        (root / (split + ".jsonl")).write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for row in built))
        files[split] = {"sha256": file_sha(root / (split + ".jsonl")), "groups_sha256": file_sha(root / (split + "-groups.json")),
                        "pairs": len(built), "independent_groups": len(originals)}
        if split == "train":
            training = built
    if freeze:
        frequencies = Counter(word for row in training for word in set(WORDS.findall(row["prompt"].casefold())))
        weights = Counter()
        histograms = {kind: defaultdict(Counter) for kind in ("prompt_length", "prompt_composition")}
        for row in training:
            weights.update(WORDS.findall(row["chosen"].casefold()))
            weights.subtract(WORDS.findall(row["rejected"].casefold()))
            for kind in histograms:
                histograms[kind][prompt_bin(row, kind)][row["label"]] += 1
        coefficient = {"SPDX-License-Identifier": "Apache-2.0", "train_sha256": files["train"]["sha256"],
                       "idf": {word: math.log((1 + len(training))/(1 + count)) + 1 for word, count in sorted(frequencies.items())},
                       "direction": {"length": 1, "tfidf_overlap": 1, "alarm_keywords": 1},
                       "answer_only_weights": {word: count for word, count in sorted(weights.items()) if count},
                       "prompt_histograms": {kind: {key: dict(value) for key, value in sorted(buckets.items())} for kind, buckets in histograms.items()}}
        scores = baselines(training, coefficient)
        coefficient["direction"] = {kind: 1 if scores[kind] >= .5 else -1 for kind in coefficient["direction"]}
        dump(BASELINE, coefficient)
    coefficient = strict_json(BASELINE.read_bytes())
    if coefficient["train_sha256"] != files["train"]["sha256"]:
        raise ValueError("immutable training-only coefficient identity mismatch")
    (root / "LICENSE").write_bytes((raw_root / "LICENSE").read_bytes())
    manifest = {"schema_version": 1, "family": FAMILY, "version": 3, "sources_sha256": file_sha(SOURCE_FILE),
                "inventory": inventory_report, "selection": selection, "files": files,
                "baseline_sha256": file_sha(BASELINE), "quarantine_sha256": file_sha(QUARANTINE),
                "normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True,
                "independent_groups": len(groups["audit"]),
                "construction": "Upstream human emotion annotations, unanimous derived valence labels and constructed compositions/preferences. No human sentiment-preference review claimed.",
                "scope": "High-agreement short English emotion-valence proxy. All Mixed texts and a subset of other labels are constructed compositions. Numeric confidence and reason correctness are not independently validated."}
    dump(root / "source-manifest.json", manifest)
    return groups, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    _, report = prepare(args.out, args.source_root, args.freeze)
    print(json.dumps(report, indent=2))
