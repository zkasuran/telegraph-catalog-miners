# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Fixed licensed source replay. No pool inputs, fitting or neural imports."""
import argparse
import hashlib
import json
import math
import shutil
import time
import unicodedata
import urllib.request
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = "TEXT_SUMMARIZATION"
TOKENIZER_SHA = "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456"
VIEWS = ("raw", "json", "prose")
LIMITS = {"train": 128, "validation": 32, "audit": 64}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def render(prompt, answer):
    return f"### Task:\n{prompt}\n\n### Answer:\n{answer}"


def view(row, style):
    result = dict(row)
    if style == "json":
        for side in ("chosen", "rejected"):
            result[side] = json.dumps({"title": row[side]}, ensure_ascii=False, separators=(",", ":"))
    elif style == "prose":
        for side in ("chosen", "rejected"):
            result[side] = "Title: " + row[side]
    elif style != "raw":
        raise ValueError("unsupported source answer format")
    return result


def fetch(url, path, expected, ceiling=700 << 20):
    if path.is_file() and not path.is_symlink() and path.stat().st_size <= ceiling:
        h = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() == expected:
            return
    started, total, h = time.monotonic(), 0, hashlib.sha256()
    temporary = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "Summary-owner-source/1"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response, temporary.open("wb") as target:
            while chunk := response.read(1 << 20):
                total += len(chunk)
                if total > ceiling or time.monotonic() - started > 600:
                    raise ValueError("source download byte or time ceiling")
                target.write(chunk)
                h.update(chunk)
        if h.hexdigest() != expected:
            raise ValueError("source hash mismatch: " + path.name)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def check_splits(manifest, splits):
    from primary import anchors
    exposure = json.loads((HERE / "exposed-audit.json").read_text())
    forbidden_dois = set(exposure["dois"])
    forbidden_entities = set(exposure["source_entity_ids"])
    groups_seen, documents_seen, texts_seen, identities_seen = set(), set(), set(), set()
    intervals = []
    for name, count in LIMITS.items():
        groups = manifest["splits"][name]
        rows = splits[name]
        if len(groups) != count or len(rows) != 2 * count:
            raise ValueError("real 128/32/64 source component floor changed")
        group_ids = {g["group_id"] for g in groups}
        documents = [d["doi"] for g in groups for d in g["documents"]]
        texts = {norm(d[f]) for g in groups for d in g["documents"] for f in ("title", "abstract")}
        identities = {i for g in groups for i in g["source_entity_ids"]}
        if set(documents) & forbidden_dois or identities & forbidden_entities:
            raise ValueError("scored original audit source/entity reused after exposure")
        if (len(group_ids) != count or len(documents) != len(set(documents)) or group_ids & groups_seen
                or set(documents) & documents_seen or texts & texts_seen or identities & identities_seen):
            raise ValueError("group, document, normalized text or author/ORCID leakage")
        if rows != [r for g in groups for r in g["pairs"]]:
            raise ValueError("source row bytes do not match the component manifest")
        for group in groups:
            date.fromisoformat(group["known_at"])
            docs = group["documents"]
            if len(docs) != 2 or len(group["pairs"]) != 2:
                raise ValueError("complete two-study source component required")
            for d in docs:
                if (d["license"] != "CC-BY-4.0" or not d["authors"] or not d["rights_clause"]
                        or not d["source_url"].startswith("https://raw.githubusercontent.com/elifesciences/elife-article-xml/")):
                    raise ValueError("primary source grant or attribution missing")
                if not {"doi:" + d["doi"], *("name:" + norm(author) for author in d["authors"])} <= set(group["source_entity_ids"]):
                    raise ValueError("primary author or DOI omitted from source grouping")
            for row in group["pairs"]:
                if row["group_id"] != group["group_id"] or row["target"] not in ("A", "B"):
                    raise ValueError("source component or target identity mismatch")
                expected = (f"Give a concise title for Study {row['target']}.\n\nStudy A:\n{docs[0]['abstract']}"
                            f"\n\nStudy B:\n{docs[1]['abstract']}")
                own, other = docs if row["target"] == "A" else reversed(docs)
                if (row["prompt"] != expected or row["chosen"] != own["title"] or row["rejected"] != other["title"]
                        or row["source_id"] != own["doi"] or row["donor_id"] != other["doi"]
                        or not row["exclusive_title_anchors"] or row["chosen"] == row["rejected"]
                        or anchors(own, other) != row["exclusive_title_anchors"]):
                    raise ValueError("literal complete source/title preference binding failed")
        groups_seen.update(group_ids)
        documents_seen.update(documents)
        texts_seen.update(texts)
        identities_seen.update(identities)
        dates = sorted(g["known_at"] for g in groups)
        intervals.append((dates[0], dates[-1]))
    if not intervals[0][1] < intervals[1][0] or not intervals[1][1] < intervals[2][0]:
        raise ValueError("constructed-task availability split is not strictly ordered")
    return {"normalized_overlap": 0, "source_entity_overlap": 0, "independent_groups": 224,
            "audit_groups": 64, "audit_pairs": 128, "task_time_ordered": True}


def read_splits(root):
    contract = json.loads((HERE / "contract.json").read_text())
    raw = (HERE / "data/manifest.json").read_bytes()
    if digest(raw) != contract["source_manifest_sha256"]:
        raise ValueError("fixed source manifest changed")
    manifest = json.loads(raw)
    splits = {}
    for name in LIMITS:
        path = root / (name + ".jsonl")
        raw = path.read_bytes()
        if digest(raw) != contract["splits"][name]["sha256"]:
            raise ValueError("fixed source split changed: " + name)
        splits[name] = [json.loads(line) for line in raw.decode().split("\n") if line]
        if any(type(row.get("margin")) not in (int, float) or not math.isfinite(row["margin"]) for row in splits[name]):
            raise ValueError("invalid source preference margin")
    boundary = check_splits(manifest, splits)
    report = {"schema_version": 1, "family": FAMILY, "scope": contract["scope"],
              "files": contract["splits"], "source_manifest_sha256": contract["source_manifest_sha256"],
              "boundary": boundary, "training_allowed": False,
              "authorship": "Primary human-published title associations; donor preferences constructed by AI-assisted code, not human preference votes."}
    return report, splits


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    for name in LIMITS:
        source, target = HERE / ("data/" + name + ".jsonl"), root / (name + ".jsonl")
        if target.is_file() and target.read_bytes() != source.read_bytes():
            raise ValueError("existing source freeze must not be overwritten")
        if not target.exists():
            shutil.copyfile(source, target)
    report, _ = read_splits(root)
    write_json(root / "data-report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args().root)))
