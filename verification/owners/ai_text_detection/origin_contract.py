# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Grouped source-provenance supervision, distinct from preference comparisons."""
import json
from common import COUNTS, identity, render, FORMATS


def origin_policy(recipe):
    expected = {"coefficient": 0.25, "human_target": 0.9, "synthetic_target": 0.1,
                "input": "equal_byte_prompt_only", "pooling": "cls", "head_exported": False}
    obj = recipe.get("origin_auxiliary")
    if obj != expected or any(type(obj[k]) not in (int, float) for k in
                              ("coefficient", "human_target", "synthetic_target")):
        raise ValueError("fixed separate training-only origin auxiliary required")
    return dict(obj)


def raw_origin_view(row):
    if row["split"] not in COUNTS or row["style"] not in FORMATS:
        raise ValueError("source-origin baseline requires supported source rows")
    text = json.loads(row["prompt"])["text"]
    if not isinstance(text, str) or not 32 <= len(text.encode()) <= 256:
        raise ValueError("source-origin text outside original byte ceiling")
    label = "Human" if row["origin"] == "human_control" else "AI-generated"
    if row["origin"] not in ("human_control", "synthetic") or row["label"] != label or identity(text) != row["text_id"]:
        raise ValueError("origin source identity or label mismatch")
    # Empty answer. Neither label words nor cue text is included in this input.
    return {"input": render(row["prompt"], ""), "text": text, "text_id": row["text_id"], "source_id": row["source_id"],
            "group_id": row["group_id"], "view_id": row["view_id"], "split": row["split"], "year": row["year"],
            "human": int(row["origin"] == "human_control"), "text_bytes": len(text.encode())}


def equal_byte_views(rows):
    if any(r["split"] not in ("train", "validation") for r in rows):
        raise ValueError("origin auxiliary excludes audit rows")
    views = [raw_origin_view(r) for r in rows]
    grouped = {}
    for r in views:
        grouped.setdefault(r["view_id"], []).append(r)
    cropped = []
    for group in grouped.values():
        if len(group) != 2 or {r["human"] for r in group} != {0, 1}:
            raise ValueError("byte-controlled origin views must remain paired")
        def boundaries(text):
            result, n = {0}, 0
            for char in text:
                n += len(char.encode())
                result.add(n)
            return result
        common = boundaries(group[0]["text"]).intersection(boundaries(group[1]["text"]))
        n = max(common)
        if not 32 <= n <= 256:
            raise ValueError("equal-byte source prefixes outside original byte ceiling")
        for r in group:
            text = r["text"].encode()[:n].decode("utf-8")
            cropped.append({**r, "input": render(json.dumps({"text": text}, ensure_ascii=False, separators=(",", ":")), ""),
                "text": text, "original_text_id": r["text_id"], "text_id": identity(text), "text_bytes": n,
                "changes": "Contiguous UTF-8-boundary prefix, equal byte count within the source pair. No score or label selects its length."})
    return cropped


def check_origin_training(rows, validation):
    if not 2 * COUNTS["train"] <= len(rows) <= 6 * COUNTS["train"] or any(r["split"] != "train" for r in rows):
        raise ValueError("origin fitting requires training rows only")
    train = equal_byte_views(rows)
    valid = equal_byte_views(validation)
    if any(r["split"] != "validation" for r in valid) or len(valid) != 2 * COUNTS["validation"]:
        raise ValueError("origin diagnostic validation roles/count changed")
    if (any(r["year"] != 2015 or r["group_id"] != identity(r["source_id"]) for r in valid) or
            len({r["group_id"] for r in valid}) != COUNTS["validation"] or
            len({r["text_id"] for r in valid}) != len(valid)):
        raise ValueError("origin diagnostic grouping/time/identity changed")
    by_view = {}
    for r in train:
        if r["group_id"] != identity(r["source_id"]) or not 2012 <= r["year"] <= 2014:
            raise ValueError("origin training source grouping/time changed")
        by_view.setdefault(r["view_id"], []).append(r)
    if (len({r["group_id"] for r in train}) != COUNTS["train"] or
            any(len(v) != 2 or {r["human"] for r in v} != {0, 1} for v in by_view.values()) or
            len({r["text_id"] for r in train}) != len(train)):
        raise ValueError("origin auxiliary must retain balanced independent article views")
    if ({r["group_id"] for r in train}.intersection(r["group_id"] for r in valid) or
            {r["text_id"] for r in train}.intersection(r["text_id"] for r in valid)):
        raise ValueError("origin training/validation normalized or article overlap")
    return train, valid


def fit_length(rows):
    if not rows or any(r["split"] != "train" for r in rows):
        raise ValueError("length baseline fits training only")
    thresholds = sorted({r["text_bytes"] for r in rows} | {0, 257})
    options = []
    for threshold in thresholds:
        for direction in (-1, 1):
            wins = sum(int((r["text_bytes"] >= threshold) == (r["human"] == (direction == 1))) for r in rows)
            options.append((wins, -threshold, direction))
    wins, neg_threshold, direction = max(options)
    return {"fit_split": "train", "threshold_bytes": -neg_threshold, "direction": direction,
            "training_correct": wins, "training_rows": len(rows), "floating_point_fit": False}


def length_accuracy(rows, fitted):
    return sum(int((r["text_bytes"] >= fitted["threshold_bytes"]) == (r["human"] == (fitted["direction"] == 1)))
               for r in rows) / len(rows)
