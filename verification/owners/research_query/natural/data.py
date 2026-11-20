# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Standalone original human preference source and immutable split contracts."""
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


def render(prompt, answer):
    return f"### Task:\n{prompt}\n\n### Answer:\n{answer}"


def original_quality(row):
    import math
    labels = row.get("labels")
    if isinstance(labels, dict):
        names, numbers, counts = (labels.get(key, []) for key in ("name", "value", "count"))
        if not len(names) == len(numbers) == len(counts):
            return None
        labels = [{"name": name, "value": value, "count": count} for name, value, count in zip(names, numbers, counts)]
    if not isinstance(labels, list):
        return None
    values = [value for value in labels if value.get("name") == "quality"]
    if len(values) != 1:
        return None
    value, count = values[0].get("value"), values[0].get("count")
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 or type(count) is not int or count < 1:
        return None
    return {"value": value, "count": count}


def check_confidence(rows):
    from collections import Counter
    totals, counts = Counter(), Counter()
    for row in rows:
        direction = len(row["chosen"]) > len(row["rejected"])
        if len(row["chosen"]) == len(row["rejected"]):
            raise ValueError("confidence buckets require distinct preferred lengths")
        diff = row["chosen_source_quality"]["value"] - row["rejected_source_quality"]["value"]
        units = 10 if diff >= .15 else 5 if diff > 0 else 3 if diff == 0 else 1
        if type(row["confidence_units"]) is not int or row["confidence_units"] != units:
            raise ValueError("original training confidence was altered")
        totals[direction] += units
        counts[direction] += 1
    if counts != {True: len(rows) // 2, False: len(rows) // 2}:
        raise ValueError("confidence-weighted source must remain length balanced")
    for row in rows:
        if row["confidence_normalizer"] != totals[len(row["chosen"]) > len(row["rejected"])] or row["confidence_bucket_size"] != len(rows) // 2:
            raise ValueError("confidence normalizer is not training-fitted")


def check_normalized_aliases(splits):
    helper = source_helper()
    text_groups = {}
    for rows in splits.values():
        for row in rows:
            for key in ("prompt", "chosen", "rejected"):
                text = helper.norm(row[key])
                if text in text_groups and text_groups[text] != row["group_id"]:
                    raise ValueError("normalized source aliases counted as independent groups")
                text_groups[text] = row["group_id"]


def source_helper():
    path = ROOT / "support/qa-experiment/data.py"
    if digest(path.read_bytes()) != "5a1e426c814d0f02afaecce3963b56ccf025c98b61cc55e2bc36da0577c3a16d":
        raise ValueError("original source eligibility/grouping code changed")
    spec = importlib.util.spec_from_file_location("natural_original_source_contract", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_splits(directory):
    report = json.loads((directory / "data-report.json").read_text())
    splits = {}
    for name, pin in report["files"].items():
        raw = (directory / (name + ".jsonl")).read_bytes()
        if len(raw) > 8 << 20 or digest(raw) != pin["sha256"]:
            raise ValueError("natural original split hash or size changed")
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != pin["rows"] or len({row["group_id"] for row in rows}) != pin["groups"]:
            raise ValueError("complete original component counts required")
        if any(row["chosen_rank"] != 0 or row["rejected_rank"] <= row["chosen_rank"] or row["source_revision"] != "fdf72ae0827c1cda404aff25b6603abec9e3399b" for row in rows):
            raise ValueError("original best-rank source preference direction changed")
        splits[name] = rows
    helper = source_helper()
    helper.check_splits(splits)
    check_normalized_aliases(splits)
    for name in ("train", "warmup"):
        check_confidence(splits[name])
    if max(report["files"][name]["date_range"][1] for name in ("train", "warmup")) >= report["files"]["validation"]["date_range"][0] or not report["files"]["validation"]["date_range"][0] < report["files"]["audit"]["date_range"][0]:
        raise ValueError("chronological source/entity cohorts overlap")
    return report, splits


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    report, splits = read_splits(HERE / "fixtures")
    for name in ("train.jsonl", "warmup.jsonl", "validation.jsonl", "audit.jsonl", "data-report.json", "frozen-controls.json"):
        shutil.copyfile(HERE / "fixtures" / name, directory / name)
    for name in ("LICENSE", "NOTICE", "KIT-LICENSE", "UPSTREAM-APACHE-2.0.txt"):
        shutil.copyfile(HERE / name, directory / name)
    # Source grants and original text/ranks are checked against the full pinned
    # original inventories before any remote model is loaded. No credential.
    import pyarrow.parquet as pq
    spec = json.loads((HERE / "sources.json").read_text())
    fetch_spec = importlib.util.spec_from_file_location("natural_source_fetcher", ROOT / "support/training/data.py")
    fetcher = importlib.util.module_from_spec(fetch_spec)
    fetch_spec.loader.exec_module(fetcher)
    original = directory / "sources"
    original.mkdir(exist_ok=True)
    token_url, token_sha = spec["files"]["tokenizer.json"]
    fetcher.fetch(token_url, original / "tokenizer.json", token_sha, 8 << 20)
    by_id = {}
    counts = {}
    columns = ["message_id", "text", "rank", "parent_id", "message_tree_id", "synthetic", "deleted", "review_result", "review_count", "labels", "lang"]
    for split in ("train", "validation"):
        url, expected = spec["files"]["oasst-" + split + ".parquet"]
        path = original / ("oasst-" + split + ".parquet")
        fetcher.fetch(url, path, expected, 50 << 20)
        rows = pq.read_table(path, columns=columns).to_pylist()
        counts[split] = len(rows)
        by_id.update({row["message_id"]: row for row in rows})
    if counts != {"train": 84437, "validation": 4401}:
        raise ValueError("full original message inventory changed")
    helper = source_helper()
    checked = 0
    for rows in splits.values():
        for row in rows:
            for text, name, rank in ((row["prompt"], "root_id", None), (row["chosen"], "chosen_id", row["chosen_rank"]), (row["rejected"], "rejected_id", row["rejected_rank"])):
                source = by_id[row[name]]
                if source["text"] != text or not helper.eligible(source) or source["message_tree_id"] != row["tree_id"] or rank is not None and source["rank"] != rank:
                    raise ValueError("original annotated preference was altered")
            if row.get("training_phase") in ("train", "warmup"):
                if original_quality(by_id[row["chosen_id"]]) != row["chosen_source_quality"] or original_quality(by_id[row["rejected_id"]]) != row["rejected_source_quality"]:
                    raise ValueError("original quality annotation was altered")
            checked += 1
    write_json(directory / "source-reverification.json", {"original_messages": counts, "checked_preference_pairs": checked,
               "grants_verified": True, "normalized_overlap": 0, "original_rank_and_text_unchanged": True,
               "human_source_annotations_not_factual_certification": True, "audit_scored": False})
    return report
