# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Full licensed annotation inventory and new author/thread/time-held-out pairs."""
import argparse
import csv
import json
import math
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from contracts import FAMILY, FROZEN, HERE, fetch, file_sha, sha, strict_json
from source_data import WORDS, features, group_id, inventory

LABELS = ("Positive", "Negative", "Neutral", "Mixed")
COUNTS = {"train": 128, "validation": 32, "audit": 32}
SOURCE_FILE = HERE / "retrain-sources.json"
BASELINE_FILE = HERE / "retrain-baseline.json"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def download(url, path, expected=None):
    if path.is_file() and (expected is None or file_sha(path) == expected):
        return
    if url not in {item["url"] for item in strict_json(SOURCE_FILE.read_bytes())["files"].values()}:
        raise ValueError("unreviewed source URL")
    start = time.monotonic()
    request = urllib.request.Request(url, headers={"User-Agent": "Sentiment-source-inventory/2.0"})
    total = 0
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        with urllib.request.urlopen(request, timeout=30) as response, temporary.open("wb") as out:
            if response.status != 200 or response.geturl() != url:
                raise ValueError("source transport mismatch")
            while chunk := response.read(1 << 20):
                total += len(chunk)
                if total > 64 << 20 or time.monotonic() - start > 300:
                    raise ValueError("source byte/time ceiling")
                out.write(chunk)
        if expected is not None and file_sha(temporary) != expected:
            raise ValueError("source hash mismatch")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_inventory(root, mapping):
    originals = {}
    counts = {}
    for part in range(1, 4):
        name = f"goemotions_{part}.csv"
        count = 0
        with (root / name).open(newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"text", "id", "author", "link_id", "created_utc", "rater_id", "neutral",
                        "example_very_unclear", *mapping["positive"], *mapping["negative"], *mapping["ambiguous"]}
            if not required <= set(reader.fieldnames or []):
                raise ValueError("raw source annotation schema mismatch")
            for row in reader:
                count += 1
                if count > 100000:
                    raise ValueError("raw inventory count ceiling")
                timestamp = float(row["created_utc"])
                if not math.isfinite(timestamp):
                    raise ValueError("nonfinite source timestamp")
                metadata = {"text": row["text"].strip(), "source_id": row["id"],
                            "author": sha(row["author"].encode()), "thread": sha(row["link_id"].encode()),
                            "timestamp": int(timestamp), "anonymous": row["author"] in ("", "[deleted]")}
                original = originals.setdefault(row["id"], {**metadata, "votes": {}, "conflict": False})
                if any(original[key] != value for key, value in metadata.items()):
                    original["conflict"] = True
                positive = any(row[name] == "1" for name in mapping["positive"])
                negative = any(row[name] == "1" for name in mapping["negative"])
                ambiguous = any(row[name] == "1" for name in mapping["ambiguous"])
                neutral = row["neutral"] == "1"
                vote = ("Mixed" if positive and negative else "Positive" if positive else
                        "Negative" if negative else "Neutral" if neutral and not ambiguous else None)
                if row["example_very_unclear"] == "True" or row["example_very_unclear"] == "1":
                    vote = None
                rater = row["rater_id"]
                if rater in original["votes"] and original["votes"][rater] != vote:
                    original["conflict"] = True
                original["votes"][rater] = vote
        counts[name] = count
    selected = []
    reasons = Counter()
    times = sorted(row["timestamp"] for row in originals.values())
    if len(times) < 1000:
        raise ValueError("incomplete raw source inventory")
    cut_train, cut_validation = times[len(times)*3//4], times[len(times)*7//8]
    for row in originals.values():
        votes = Counter(value for value in row["votes"].values() if value is not None)
        if row["conflict"] or row["anonymous"] or not 8 <= len(row["text"].encode()) <= 128:
            reasons["conflict_anonymous_or_length"] += 1
            continue
        if not votes:
            reasons["no_valence_vote"] += 1
            continue
        label, n = votes.most_common(1)[0]
        if n < 2 or n * 2 <= len(row["votes"]) or label == "Mixed":
            reasons["no_strict_majority_or_direct_mixed"] += 1
            continue
        split = ("train" if row["timestamp"] < cut_train else
                 "validation" if row["timestamp"] < cut_validation else "audit")
        selected.append({key: row[key] for key in ("text", "source_id", "author", "thread", "timestamp")} |
                        {"label": label, "split": split, "group_id": group_id(row["text"])})
    return selected, {"raw_rows": counts, "distinct_comments": len(originals),
                      "time_cutoffs": [cut_train, cut_validation], "excluded": dict(reasons)}


def grouped_select(rows, excluded_groups):
    memberships = defaultdict(set)
    for row in rows:
        for key in ("author", "thread", "group_id"):
            memberships[key, row[key]].add(row["split"])
    eligible = defaultdict(list)
    removed = Counter()
    for row in rows:
        if row["group_id"] in excluded_groups:
            removed["prior_source_overlap"] += 1
        elif any(len(memberships[key, row[key]]) != 1 for key in ("author", "thread", "group_id")):
            removed["cross_time_author_thread_or_text"] += 1
        else:
            eligible[row["split"], row["label"]].append(row)
    used = set()
    splits = {}
    for split, per_label in COUNTS.items():
        pool = {}
        for label in LABELS[:3]:
            need = per_label * (2 if label != "Neutral" else 1)
            pool[label] = []
            for row in sorted(eligible[split, label], key=lambda r: sha(("sentiment-v2:" + r["group_id"]).encode())):
                keys = {(key, row[key]) for key in ("author", "thread", "group_id")}
                if used & keys:
                    continue
                used.update(keys)
                pool[label].append(row)
                if len(pool[label]) == need:
                    break
            if len(pool[label]) != need:
                raise ValueError(f"insufficient independent {split} {label} groups: {len(pool[label])}/{need}")
        chosen = []
        for label in LABELS[:3]:
            for row in pool[label][:per_label]:
                chosen.append({**row, "sources": [row["source_id"]], "entities": [row["author"]],
                               "threads": [row["thread"]], "source_groups": [row["group_id"]],
                               "construction": "strict-majority annotation valence mapping"})
        for positive, negative in zip(pool["Positive"][per_label:], pool["Negative"][per_label:]):
            text = positive["text"] + " However, " + negative["text"]
            chosen.append({"text": text, "label": "Mixed", "split": split, "group_id": group_id(text),
                           "sources": [positive["source_id"], negative["source_id"]],
                           "entities": [positive["author"], negative["author"]],
                           "threads": [positive["thread"], negative["thread"]],
                           "source_groups": [positive["group_id"], negative["group_id"]],
                           "construction": "constructed juxtaposition of independently annotated opposite-valence comments"})
        splits[split] = chosen
    check_groups(splits)
    return splits, {"eligible": {f"{split}/{label}": len(values) for (split, label), values in sorted(eligible.items())},
                    "removed": dict(removed)}


def check_groups(splits):
    used = set()
    for split, rows in splits.items():
        for row in rows:
            if row["label"] not in LABELS or row["group_id"] != group_id(row["text"]):
                raise ValueError("source identity or label mismatch")
            keys = {(key, value) for key in ("sources", "entities", "threads", "source_groups") for value in row[key]}
            keys.add(("constructed_text", row["group_id"]))
            if keys & used:
                raise ValueError("source/entity/thread/normalized overlap")
            used.update(keys)
        if Counter(r["label"] for r in rows) != {label: COUNTS[split] for label in LABELS}:
            raise ValueError("unbalanced independent source split")


def make_pairs(rows, style=None, training=False):
    pairs = []
    for label in LABELS:
        selected = sorted((r for r in rows if r["label"] == label), key=lambda r: r["group_id"])
        for index, row in enumerate(selected):
            text = row["text"]
            prompt_style = index % 4
            prompt = (json.dumps({"text": text}, ensure_ascii=False, separators=(",", ":")) if prompt_style == 0 else
                      text if prompt_style == 1 else "Classify the sentiment of this text: " + text if prompt_style == 2 else
                      "Is the following text Positive, Negative, Neutral or Mixed?\n" + text)
            selected_style = (2 if training and index % 20 == 0 else index % 2) if style is None else style
            if selected_style not in (0, 1, 2):
                raise ValueError("unsupported answer format")

            def answer(value):
                if selected_style == 2:
                    return value
                if selected_style == 1:
                    return f"Sentiment: {value}. Text assessed: {text}"
                return json.dumps({"sentiment": value, "reason": "Text assessed: " + text},
                                  ensure_ascii=False, separators=(",", ":"))
            for wrong in LABELS:
                if wrong != label:
                    pairs.append({"prompt": prompt, "chosen": answer(label), "rejected": answer(wrong),
                                  "label": label, "wrong_label": wrong, "group_id": row["group_id"],
                                  "label_kind": "preference_pair", "margin": 0.0})
    return pairs


def baselines(rows, coefficient):
    differences = [features(row, coefficient) for row in rows]
    result = {key: sum(.5 if abs(d[key]) < 1e-10 else float(d[key] * direction > 0)
                      for d in differences) / len(rows) for key, direction in coefficient["direction"].items()}
    weights = coefficient["answer_only_weights"]
    def answer_score(text):
        return sum(weights.get(word, 0) * count for word, count in Counter(WORDS.findall(text.casefold())).items())
    result["answer_only"] = sum(.5 if answer_score(r["chosen"]) == answer_score(r["rejected"]) else
                                float(answer_score(r["chosen"]) > answer_score(r["rejected"])) for r in rows) / len(rows)
    return result


def prepare(root, freeze=False):
    root.mkdir(parents=True, exist_ok=True)
    sources = strict_json(SOURCE_FILE.read_bytes())
    if not freeze and sources["stage"] != "verified":
        raise ValueError("complete source grants and hashes required before remote training")
    for name, spec in sources["files"].items():
        download(spec["url"], root / name, spec.get("sha256"))
        if freeze:
            spec.update(sha256=file_sha(root / name), bytes=(root / name).stat().st_size)
    license_text = (root / "LICENSE").read_text()
    if "copyright license to reproduce, prepare Derivative Works of," not in license_text:
        raise ValueError("original Apache grant missing")
    mapping = strict_json((root / "sentiment_mapping.json").read_bytes())
    raw, inventory_report = read_inventory(root, mapping)
    previous = root / "excluded-prior-uci.zip"
    fetch(FROZEN["source_url"], previous, FROZEN["source_sha256"], 1 << 20)
    old_rows, _ = inventory(previous)
    splits, group_report = grouped_select(raw, {group_id(r["text"]) for r in old_rows})
    files = {}
    training = None
    for split, rows in splits.items():
        dump(root / (split + "-groups.json"), rows)
        pairs = make_pairs(rows, training=split == "train")
        text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for r in pairs)
        path = root / (split + ".jsonl")
        path.write_text(text)
        files[split] = {"sha256": file_sha(path), "pairs": len(pairs), "independent_groups": len(rows),
                        "groups_sha256": file_sha(root / (split + "-groups.json"))}
        if split == "train":
            training = pairs
    if freeze:
        sources["stage"] = "verified"
        sources["grant"] = "Apache-2.0 section 2 and retained section 4 attribution"
        dump(SOURCE_FILE, sources)
        frequencies = Counter(word for r in training for word in set(WORDS.findall(r["prompt"].casefold())))
        weights = Counter()
        for row in training:
            weights.update(WORDS.findall(row["chosen"].casefold()))
            weights.subtract(WORDS.findall(row["rejected"].casefold()))
        coefficient = {"SPDX-License-Identifier": "Apache-2.0", "train_sha256": files["train"]["sha256"],
                       "idf": {w: math.log((1 + len(training))/(1 + n)) + 1 for w, n in sorted(frequencies.items())},
                       "answer_only_weights": {w: n for w, n in sorted(weights.items()) if n},
                       "direction": {"length": 1, "tfidf_overlap": 1, "alarm_keywords": 1}}
        scores = baselines(training, coefficient)
        coefficient["direction"] = {k: 1 if scores[k] >= .5 else -1 for k in coefficient["direction"]}
        dump(BASELINE_FILE, coefficient)
    coefficient = strict_json(BASELINE_FILE.read_bytes())
    if coefficient["train_sha256"] != files["train"]["sha256"]:
        raise ValueError("train-fitted immutable baseline identity mismatch")
    manifest = {"schema_version": 1, "family": FAMILY, "source": "GoEmotions", "sources_sha256": file_sha(SOURCE_FILE),
                "inventory": inventory_report, "group_selection": group_report, "files": files,
                "baseline_sha256": file_sha(BASELINE_FILE), "normalized_overlap": 0,
                "independent_groups": len(splits["audit"]), "audit_used_for_selection": False, "grants_verified": True,
                "construction": "Original human emotion annotations, deterministic valence mapping and constructed Mixed juxtapositions. Preferences are AI-assisted programmatic work, not human-reviewed.",
                "scope": "Four-way English emotion-valence proxy. Mixed composites are constructed, not natural human-reviewed Mixed labels."}
    dump(root / "source-manifest.json", manifest)
    return splits, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freeze-sources", action="store_true")
    args = parser.parse_args()
    _, report = prepare(args.out, args.freeze_sources)
    print(json.dumps(report, indent=2))
