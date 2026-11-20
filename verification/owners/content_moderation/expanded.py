# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pure full-inventory source grouping and frozen M2 plan contracts."""
import json
import re
from collections import Counter

from contracts import FAMILY, HERE, sha

SPLIT = {"train": 1, "validation": 2, "test": 4}
LABEL = {"Safe": 1, "Flagged": 2, None: 4}
MAX_ROWS = 2200000


def plan():
    value = json.loads((HERE / "expanded.json").read_bytes())
    validate_plan(value)
    return value


def validate_plan(value):
    expected = {"schema_version": 1, "family": FAMILY, "arm": "M2", "epochs": 2, "train_groups": 1536,
                "validation_groups": 256, "audit_groups": 256, "job_minutes": 330, "max_source_bytes": 320,
                "max_tokens": 256, "exclusions_count": 512, "source_id": "google/civil_comments",
                "source_revision": "f2970eb3a55777454c94069077cc8d9b5866312d"}
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("frozen M2 source, audit or CPU contract changed")
    hp = {"seed": 42, "lora_rank": 8, "lora_alpha": 32, "lr": 0.00001, "lora_dropout": 0.1, "weight_decay": 0.01,
          "batch_size": 2, "grad_accum": 8, "max_length": 256, "early_stopping": True, "es_min_delta": 0.001, "es_patience": 1}
    if value.get("hyperparameters") != hp:
        raise ValueError("M2 frozen model/LR/source-length recipe changed")
    files = value.get("files", [])
    names = {"data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet", "data/validation-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"}
    if len(files) != 4 or {r.get("path") for r in files} != names:
        raise ValueError("complete two-shard training inventory required")
    for row in files:
        if not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")) or type(row.get("bytes")) is not int or not 0 < row["bytes"] <= 250 << 20 or row.get("split") not in SPLIT:
            raise ValueError("source file hash/size/split missing")
    for key in ("checkpoint_sha256", "config_sha256", "untrained_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", value.get("warm_start", {}).get(key, "")):
            raise ValueError("licensed model inputs unbound")
    if not re.fullmatch(r"[0-9a-f]{64}", value.get("exclusions_sha256", "")):
        raise ValueError("entire exposed audit exclusion set unbound")


def grouped_sources(rows, excluded, group_id, maximum_bytes=320):
    """Scan every row before selection, retaining conflicts even for non-extremes."""
    states, candidates = {}, {}
    counts, scanned = Counter(), 0
    for row in rows:
        scanned += 1
        if scanned > MAX_ROWS:
            raise ValueError("complete source scan ceiling")
        split, label, text = row["split"], row["label"], row["text"].strip()
        if split not in SPLIT or label not in LABEL:
            raise ValueError("unrecognized source split or annotation class")
        counts[split] += 1
        group = group_id(text)
        states[group] = states.get(group, 0) | (SPLIT[split] << 3) | LABEL[label]
        if group in excluded or label is None or not 8 <= len(text.encode()) <= maximum_bytes:
            continue
        candidate = {"text": text, "label": label, "split": split, "group_id": group,
                     "source_id": group, "domain": "civil_comments"}
        previous = candidates.get(group)
        if previous is None or text.encode() < previous["text"].encode():
            candidates[group] = candidate
    clean = [row for group, row in candidates.items() if states[group] == ((SPLIT[row["split"]] << 3) | LABEL[row["label"]])]
    return clean, {"raw_rows": scanned, "by_official_split": dict(counts), "normalized_groups": len(states),
                   "eligible_clean_groups": len(clean), "excluded_audit_groups": len(excluded),
                   "conflicting_or_cross_split_candidate_groups": len(candidates) - len(clean)}


def select_groups(rows, sizes, salts, eligible):
    result, rejected = {}, Counter()
    for split, count in sizes.items():
        official = "test" if split == "audit" else split
        selected = []
        for label in ("Safe", "Flagged"):
            candidates = sorted((r for r in rows if r["split"] == official and r["label"] == label),
                                key=lambda r: sha((salts[split] + ":" + r["group_id"]).encode()))
            chosen = []
            for row in candidates:
                if not eligible(row, split):
                    rejected[split] += 1
                    continue
                chosen.append(row)
                if len(chosen) == count // 2:
                    break
            if len(chosen) != count // 2:
                raise ValueError("insufficient independent source groups: " + split + " " + label)
            selected.extend(chosen)
        result[split] = sorted(selected, key=lambda r: r["group_id"])
    seen = set()
    for split, rows in result.items():
        ids = {r["group_id"] for r in rows}
        if len(ids) != len(rows) or seen & ids:
            raise ValueError("selected source/entity overlap")
        seen.update(ids)
    return result, dict(rejected)
