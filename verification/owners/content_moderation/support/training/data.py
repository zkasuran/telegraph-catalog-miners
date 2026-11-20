# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-bound preference pairs. No evaluation-pool inputs or model calls."""
import argparse
import hashlib
import json
import math
import re
import time
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = json.loads((HERE / "sources.json").read_text())
LIMITS = {"train": 768, "validation": 128, "test": 256}
MAX_TEXT_BYTES = 256
MAX_DOWNLOAD_BYTES = 700 << 20
WORDS = re.compile(r"[a-z0-9']+")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def group_id(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return digest(normalized.encode())


def fetch(url, path, expected, ceiling=MAX_DOWNLOAD_BYTES):
    if path.is_file() and path.stat().st_size <= ceiling:
        h = hashlib.sha256()
        with path.open("rb") as cached:
            for chunk in iter(lambda: cached.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() == expected:
            return
    partial = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "Telegraph-training-check/1.0"})
    started = time.monotonic()
    h, total = hashlib.sha256(), 0
    try:
        with urllib.request.urlopen(request, timeout=45) as response, partial.open("wb") as out:
            while chunk := response.read(1 << 20):
                total += len(chunk)
                if total > ceiling or time.monotonic() - started > 600:
                    raise ValueError("source download exceeded byte or time ceiling")
                h.update(chunk)
                out.write(chunk)
        if h.hexdigest() != expected:
            raise ValueError("source hash mismatch: " + path.name)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def sentiment(root):
    spec = SOURCES["SENTIMENT_ANALYSIS"]
    archive = root / "sentiment.zip"
    fetch(spec["url"], archive, spec["sha256"], 1 << 20)
    rows = []
    with zipfile.ZipFile(archive) as source:
        for domain, filename in (("amazon", "amazon_cells"), ("imdb", "imdb"), ("yelp", "yelp")):
            member = "sentiment labelled sentences/" + filename + "_labelled.txt"
            if source.getinfo(member).file_size > 1 << 20:
                raise ValueError("oversized source member")
            # The text can contain Unicode line-separator characters. Only LF
            # terminates a TSV source record; an empty physical line is not a row.
            for number, line in enumerate(source.read(member).decode("utf-8").split("\n"), 1):
                if not line.strip():
                    continue
                line = line.rstrip("\r")
                text, label = line.rsplit("\t", 1)
                if label not in ("0", "1"):
                    raise ValueError("invalid source sentiment label")
                group = group_id(text)
                split = "test" if domain == "yelp" else "validation" if int(group[:8], 16) % 8 == 0 else "train"
                rows.append({"text": text, "label": "Positive" if label == "1" else "Negative",
                             "split": split, "domain": domain, "source_id": f"{domain}:{number}"})
    return rows


def moderation(root):
    from pyarrow import parquet
    spec = SOURCES["CONTENT_MODERATION"]
    rows = []
    for split, (filename, sha256) in spec["files"].items():
        path = root / (split + ".parquet")
        url = f"https://huggingface.co/datasets/{spec['id']}/resolve/{spec['revision']}/{filename}"
        fetch(url, path, sha256, 250 << 20)
        seen = Counter()
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=2048, columns=["text", "toxicity"]):
            for item in batch.to_pylist():
                value, text = item["toxicity"], item["text"]
                if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("invalid toxicity fraction")
                label = "Safe" if value == 0 else "Flagged" if value >= 0.8 else None
                if label is None or not 8 <= len(text.encode()) <= MAX_TEXT_BYTES:
                    continue
                # Bounded deterministic reservoir pool, not just a majority-class prefix.
                if seen[label] >= LIMITS[split] * 8:
                    continue
                seen[label] += 1
                rows.append({"text": text, "label": label, "split": split,
                             "domain": "civil_comments", "source_id": group_id(text)})
            if seen["Safe"] >= LIMITS[split] * 8 and seen["Flagged"] >= LIMITS[split] * 8:
                break
    return rows


def select(rows):
    groups = defaultdict(list)
    for row in rows:
        text = row["text"].strip()
        if not 8 <= len(text.encode()) <= MAX_TEXT_BYTES:
            continue
        groups[group_id(text)].append({**row, "text": text})
    buckets = defaultdict(list)
    removed = 0
    for group, occurrences in groups.items():
        # Remove cross-split originals and contradictory source labels entirely.
        if len({r["split"] for r in occurrences}) != 1 or len({r["label"] for r in occurrences}) != 1:
            removed += len(occurrences)
            continue
        row = occurrences[0]
        buckets[row["split"], row["label"]].append({**row, "group_id": group})
    result = {}
    for split, count in LIMITS.items():
        labels = sorted(label for s, label in buckets if s == split)
        if len(labels) != 2:
            raise ValueError("split must contain both supported labels: " + split)
        selected = []
        for label in labels:
            candidates = sorted(buckets[split, label], key=lambda r: digest(("select-v1:" + r["group_id"]).encode()))
            if len(candidates) < count // 2:
                raise ValueError(f"not enough {split} {label} sources: {len(candidates)}")
            selected.extend(candidates[:count // 2])
        result[split] = sorted(selected, key=lambda r: r["group_id"])
    return result, removed


def pair(family, row, style=None):
    text, label = row["text"], row["label"]
    opposite = ({"Positive": "Negative", "Negative": "Positive"} if family == "SENTIMENT_ANALYSIS"
                else {"Safe": "Flagged", "Flagged": "Safe"})[label]
    field = "sentiment" if family == "SENTIMENT_ANALYSIS" else "verdict"
    # One template per original. Both answers quote the exact same source text.
    style = int(row["group_id"][:2], 16) % 2 if style is None else style
    def answer(verdict):
        if style == 2:
            return verdict
        if style == 1:
            return f"{field.capitalize()}: {verdict}. Text assessed: {text}"
        value = {field: verdict, "reason": "Text assessed: " + text}
        if family == "CONTENT_MODERATION":
            value["flagged"] = verdict == "Flagged"
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return {"prompt": json.dumps({"text": text}, ensure_ascii=False, separators=(",", ":")),
            "chosen": answer(label), "rejected": answer(opposite), "label_kind": "preference_pair",
            "label": label, "error_family": "label_flip", "group_id": row["group_id"],
            "source_id": row["source_id"], "domain": row["domain"], "style": style}


def check_splits(splits):
    seen = set()
    for split, rows in splits.items():
        for row in rows:
            text = json.loads(row["prompt"]).get("text")
            if not isinstance(text, str) or row["group_id"] != group_id(text):
                raise ValueError("source-group identity does not match original text")
        ids = [r["group_id"] for r in rows]
        if not rows or len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError("empty split or overlapping source groups: " + split)
        seen.update(ids)
        counts = Counter(r["label"] for r in rows)
        if len(counts) != 2 or len(set(counts.values())) != 1:
            raise ValueError("unbalanced labels: " + split)
        if any(r["chosen"] == r["rejected"] for r in rows):
            raise ValueError("identical pair answers")


def shortcut_baselines(splits):
    """Fit IDF and rule direction on train only, then freeze them for evaluation."""
    documents = [Counter(WORDS.findall(r["prompt"].casefold())) for r in splits["train"]]
    frequencies = Counter(word for doc in documents for word in doc)
    idf = {word: math.log((1 + len(documents)) / (1 + n)) + 1 for word, n in frequencies.items()}
    alarm = {"negative", "flagged", "danger", "hate", "threat", "kill", "harm"}
    def vector(text):
        return {w: n * idf[w] for w, n in Counter(WORDS.findall(text.casefold())).items() if w in idf}
    def cosine(a, b):
        denominator = math.sqrt(sum(v*v for v in a.values()) * sum(v*v for v in b.values()))
        return sum(v * b.get(w, 0) for w, v in a.items()) / denominator if denominator else 0.0
    def differences(row):
        prompt, chosen, rejected = vector(row["prompt"]), vector(row["chosen"]), vector(row["rejected"])
        def keywords(text):
            return sum(w in alarm for w in WORDS.findall(text.casefold()))
        return {"length": len(row["chosen"]) - len(row["rejected"]),
                "tfidf_overlap": cosine(prompt, chosen) - cosine(prompt, rejected),
                "alarm_keywords": keywords(row["chosen"]) - keywords(row["rejected"])}
    values = {s: [differences(r) for r in rows] for s, rows in splits.items()}
    def accuracy(rows, name, direction):
        return sum(0.5 if abs(r[name]) < 1e-10 else float(r[name] * direction > 0) for r in rows) / len(rows)
    direction = {name: 1 if accuracy(values["train"], name, 1) >= 0.5 else -1 for name in values["train"][0]}
    return {"train_fitted_direction": direction, "by_split": {
        split: {name: round(accuracy(rows, name, d), 4) for name, d in direction.items()}
        for split, rows in values.items()}}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def decode_jsonl(raw):
    # JSONL uses LF, not Python's additional Unicode line boundaries.
    return [json.loads(line) for line in raw.decode("utf-8").split("\n") if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=["SENTIMENT_ANALYSIS", "CONTENT_MODERATION"])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    selected, removed = select(sentiment(args.out) if args.family == "SENTIMENT_ANALYSIS" else moderation(args.out))
    splits = {s: [pair(args.family, r) for r in rows] for s, rows in selected.items()}
    check_splits(splits)
    baselines = shortcut_baselines(splits)
    if any(v > 0.65 for row in baselines["by_split"].values() for v in row.values()):
        raise ValueError("a train-fitted shortcut exceeds the 65% data gate")
    files = {}
    for split, rows in splits.items():
        path = args.out / (split + ".jsonl")
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        files[split] = {"sha256": digest(path.read_bytes()), "pairs": len(rows),
                        "labels": dict(Counter(r["label"] for r in rows))}
    report = {"family": args.family, "sources": SOURCES[args.family], "files": files, "shortcut_baselines": baselines,
              "removed_conflicts_or_cross_split": removed, "normalized_source_overlap": 0,
              "construction": "Source labels are original; preference responses and label flips are programmatically constructed. Not human-authored preference pairs.",
              "selection": "Fixed hash-order balanced subset of bounded source pool; not a population benchmark."}
    write_json(args.out / "data-report.json", report)
    print(json.dumps({"family": args.family, "files": files, "normalized_source_overlap": 0}))


if __name__ == "__main__":
    main()
