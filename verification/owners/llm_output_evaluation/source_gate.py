# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Full public source inventory and intact-byte yield. No model imports."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time
import unicodedata
import urllib.request

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "sources.json").read_text())
MAX_DOWNLOAD = 32 << 20
MAX_EXPANDED = 256 << 20
MAX_LINE = 2 << 20


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def normal(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


def fetch(url, path, expected=None):
    if path.is_file() and expected and digest(path.read_bytes()) == expected:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "Telegraph-owner-source-audit/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(MAX_DOWNLOAD + 1)
    if len(raw) > MAX_DOWNLOAD:
        raise ValueError("download ceiling")
    if expected and digest(raw) != expected:
        raise ValueError("source SHA-256 mismatch: " + path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def rows(path):
    total = 0
    with gzip.open(path, "rb") as stream:
        while True:
            line = stream.readline(MAX_LINE + 1)
            if not line:
                break
            total += len(line)
            if len(line) > MAX_LINE or total > MAX_EXPANDED:
                raise ValueError("source expansion ceiling")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("source row is not an object")
            yield value


def inventory(root):
    report = {"scope": "All four pinned HelpSteer2 payloads, not a first-page sample", "files": {}}
    intact = []
    for name, sha in PINS["files"].items():
        path = root / name
        fetch(f"https://huggingface.co/datasets/{PINS['dataset']}/resolve/{PINS['revision']}/{name}", path, sha)
        count, short, labels = 0, 0, Counter()
        for index, row in enumerate(rows(path)):
            count += 1
            outputs = [row[k] for k in ("response", "response_1", "response_2") if k in row]
            if not outputs or not isinstance(row.get("prompt"), str):
                raise ValueError("source schema changed")
            fit = len(row["prompt"].encode()) <= 256 and all(len(o.encode()) <= 256 for o in outputs)
            if fit:
                short += 1
                if name in ("train.jsonl.gz", "validation.jsonl.gz"):
                    labels[str(row["helpfulness"])] += 1
                    intact.append({**row, "source_file": name, "source_row": index + 1,
                                   "group_id": digest(normal(row["prompt"]).encode()),
                                   "annotation_kind": "source-human-attribute-rating",
                                   "response_kind": "source-model-generated", "license": "CC-BY-4.0"})
        report["files"][name] = {"sha256": digest(path.read_bytes()), "bytes": path.stat().st_size,
                                  "rows": count, "intact_256_byte_task_and_outputs": short,
                                  "intact_helpfulness_histogram": dict(labels)}
    root.mkdir(parents=True, exist_ok=True)
    (root / "intact.jsonl").write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in intact))
    report["intact_originals"] = len(intact)
    report["no_truncation"] = True
    report["model_operations"] = False
    report["snapshot_time_unix"] = int(time.time())
    write(root / "source-inventory.json", report)
    fetch(f"https://huggingface.co/datasets/{PINS['dataset']}/raw/{PINS['revision']}/README.md", root / "HELPSTEER2-CARD.md")
    fetch(PINS["grant_url"], root / "CC-BY-4.0.html")
    fetch("https://devnode.telegraphprotocol.com/engine/v1/intents/LLM_OUTPUT_EVALUATION_GENERAL", root / "live-schema.json")
    print(json.dumps(report, sort_keys=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE / "evidence/source")
    inventory(parser.parse_args().root)
