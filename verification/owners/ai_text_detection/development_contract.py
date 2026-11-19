# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pure source/data contracts. No model libraries or floating-point fitting."""
from collections import Counter
import json

from common import COUNTS, identity, lines, pair, sha
from reference_data_split import split_pairs


def validate_locked(raw, source, producer, lineage):
    if (sha(raw) != lineage["data_sha256"] or sha(raw) != producer["output_sha256"] or
            producer["run_id"] != lineage["producer_run_id"] or producer["commit"] != lineage["producer_commit"] or
            producer["source_manifest_sha256"] != lineage["source_manifest_sha256"] or
            producer["generator"] != source["generators"]["gpt2"]):
        raise ValueError("locked development producer/source/byte binding mismatch")
    records = lines(raw)
    sources = {r["source_id"]: r for split in ("train", "validation") for r in source["source_report"]["articles"][split]}
    grouped = {}
    for row in records:
        original = sources.get(row.get("source_id"))
        if (not original or row.get("split") != original["split"] or row.get("group_id") != original["group_id"] or
                row.get("source_sha256") != original["sha256"] or row.get("year") != original["year"]):
            raise ValueError("locked development source or split role changed")
        if row["origin"] == "human_control":
            if row["text"] != original["human_text"]:
                raise ValueError("human source excerpt changed")
        elif row["origin"] == "synthetic":
            if (row.get("generator") != "gpt2" or row.get("generator_revision") != producer["generator"]["revision"] or
                    row.get("generator_weight_sha256") != producer["generator"]["weight_sha256"] or
                    row.get("seed") != int(row["group_id"][:8], 16)):
                raise ValueError("synthetic producer identity changed")
        else:
            raise ValueError("invalid origin record")
        grouped.setdefault(row["source_id"], []).append(row)
    if set(grouped) != set(sources) or any(len(rows) != 2 or {r["origin"] for r in rows} != {"human_control", "synthetic"} for rows in grouped.values()):
        raise ValueError("missing or duplicated locked source origin")
    if len(records) != 2 * (COUNTS["train"] + COUNTS["validation"]):
        raise ValueError("locked development record count changed")
    return records


def training_pairs(original, bare_group_count=1):
    # One whole article group gives one distinct opposite bare string per label.
    # This repairs repeated-negative leakage without changing the primary gate.
    bare = set(sorted({r["group_id"] for r in original if r["split"] == "train"})[:bare_group_count])
    return [pair(r, "bare_label" if r["group_id"] in bare else
                 "json" if int(r["group_id"][:2], 16) % 2 == 0 else "prose")
            for r in original if r["split"] == "train"]


def repetition_gate(rows, seed=42, eval_fraction=0.15):
    """Actual primary split function and literal stripped-negative expressions.

    Only deterministic counting is local. The full upstream surface model gate
    still runs in the remote workflow, unchanged.
    """
    if not rows:
        raise ValueError("no data rows")
    counts = Counter(r["rejected"].strip() for r in rows)
    train, held = split_pairs(rows, eval_fraction, seed)
    seen = {r["rejected"].strip() for r in train}
    leak = sum(r["rejected"].strip() in seen for r in held) / len(held) if held else 0
    unique, top = len(counts) / len(rows), max(counts.values()) / len(rows)
    return {"pairs": len(rows), "internal_holdout": len(held), "unique_rejected_frac": unique,
            "top_rejected_share": top, "eval_negative_leak": leak,
            "passed": unique >= 0.80 and top <= 0.02 and leak <= 0.10}
