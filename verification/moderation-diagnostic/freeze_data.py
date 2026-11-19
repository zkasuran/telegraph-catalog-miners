# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Freeze source-only audit identities before any model comparison."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from core import (
    CONTRACT,
    audit_select,
    load_shared,
    remote_only,
    sha,
    validate_exclusions,
    write,
)


def freeze(root, extra):
    data, _, _ = load_shared()
    original = root / "originals"
    selected, removed = data.select(data.moderation(original))
    splits = {s: [data.pair("CONTENT_MODERATION", r) for r in rows] for s, rows in selected.items()}
    data.check_splits(splits)
    excluded = validate_exclusions(extra)
    files = {}
    for split, rows in splits.items():
        raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        if sha(raw) != CONTRACT["original_splits"][split]:
            raise ValueError("original source split changed: " + split)
        (original / (split + ".jsonl")).write_bytes(raw)
        files[split] = {"sha256": sha(raw), "n": len(rows), "labels": dict(Counter(r["label"] for r in rows))}
        excluded.update(r["group_id"] for r in rows)
    baselines = data.shortcut_baselines({k: splits[k] for k in ("train", "validation")})
    if any(v > 0.65 for row in baselines["by_split"].values() for v in row.values()):
        raise ValueError("source shortcut gate failed")
    spec = data.SOURCES["CONTENT_MODERATION"]
    from pyarrow import parquet
    def official_rows():
        scanned = 0
        for batch in parquet.ParquetFile(original / "test.parquet").iter_batches(batch_size=2048, columns=["text", "toxicity"]):
            for item in batch.to_pylist():
                scanned += 1
                if scanned > 300000:
                    raise ValueError("official audit scan ceiling")
                text, value = item["text"], item["toxicity"]
                if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("invalid official toxicity")
                text = text.strip()
                if not 8 <= len(text.encode()) <= data.MAX_TEXT_BYTES:
                    continue
                label = "Safe" if value == 0 else "Flagged" if value >= 0.8 else None
                group = data.group_id(text)
                yield {"text": text, "label": label, "group_id": group, "source_id": group,
                       "domain": "civil_comments", "split": "test"}
    audit, conflicts = audit_select(official_rows(), excluded)
    audit_pairs = [data.pair("CONTENT_MODERATION", r) for r in audit]
    data.check_splits({"train": splits["train"], "validation": splits["validation"], "audit": audit_pairs})
    audit_baselines = data.shortcut_baselines({"train": splits["train"], "audit": audit_pairs})
    if any(v > 0.65 for v in audit_baselines["by_split"]["audit"].values()):
        raise ValueError("audit source shortcut gate failed")
    raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in audit_pairs).encode()
    private = root / "audit-private"
    private.mkdir()
    (private / "audit.jsonl").write_bytes(raw)
    exclusion_raw = json.dumps(sorted(excluded), separators=(",", ":")).encode()
    manifest = {"contract_sha256": sha((Path(__file__).parent / "contract.json").read_bytes()),
                "source": spec, "original_files": files, "original_removed_occurrences": removed,
                "source_baselines": baselines, "audit_baselines": audit_baselines,
                "audit_sha256": sha(raw), "audit_originals": 256,
                "audit_labels": dict(Counter(r["label"] for r in audit_pairs)),
                "audit_groups": [r["group_id"] for r in audit_pairs],
                "exclusion_sha256": sha(exclusion_raw), "exclusion_count": len(excluded),
                "extra_exclusions_sha256": sha(json.dumps(sorted(extra)).encode()),
                "conflicting_audit_groups_removed": conflicts,
                "selection": "Full pinned official test-file scan, normalized-group conflict removal, fixed salted hash order, 128 per label.",
                "salt": CONTRACT["audit"]["salt"], "frozen_before_model_comparison": True,
                "scope": spec["scope"], "pretraining_overlap": "unknown"}
    write(root / "evidence" / "exclusions.json", sorted(excluded))
    write(root / "evidence" / "audit-freeze.json", manifest)
    return manifest


if __name__ == "__main__":
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    import os
    extra = json.loads(os.environ.get("CAMPAIGN_EXCLUSIONS_JSON", "[]"))
    (args.root / "originals").mkdir(parents=True)
    freeze(args.root, extra)
