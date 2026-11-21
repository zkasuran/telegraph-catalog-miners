# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded constructed views of training originals, never an independent audit."""
from collections import Counter
from fractions import Fraction
import json
import re

from data import FORMATS, RATINGS, format_pair
from source_gate import digest
from presentation import VERSION as PRESENTATION_VERSION, evaluation_view

RECIPES = ("json-original-v1", "format-transfer-v1", "format-transfer-v2", "format-transfer-v3")
BARE_PER_LABEL = 3
MAX_ORIGINALS = 160
GRAD_ACCUM = 16


def pair_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows).encode()


def view_statistics(rows):
    if not rows:
        raise ValueError("empty training views")
    negatives = Counter(row["rejected"].strip() for row in rows)
    longer = sum(len(row["chosen"]) > len(row["rejected"]) for row in rows)
    ties = sum(len(row["chosen"]) == len(row["rejected"]) for row in rows)
    length = Fraction(2 * longer + ties, 2 * len(rows))
    return {"pairs": len(rows), "unique_negative_count": len(negatives),
            "largest_negative_count": max(negatives.values()),
            "unique_negative_share": float(Fraction(len(negatives), len(rows))),
            "largest_negative_share": float(Fraction(max(negatives.values()), len(rows))),
            "length_shortcut_accuracy": float(max(length, 1 - length)),
            "formats": dict(Counter(row.get("format", "json") for row in rows)),
            "label_counts": dict(Counter(row["label"] for row in rows)),
            "wrong_label_counts": dict(Counter(row["wrong_label"] for row in rows)),
            "independent_source_groups": len({row["group_id"] for row in rows}),
            "surface_fit_run": False, "full_primary_gate_passed": False}


def build_recipe(train, name, source_train_sha256, baseline_sha256):
    if name not in RECIPES or any(not isinstance(sha, str) or re.fullmatch(r"[a-f0-9]{64}", sha) is None
                                  for sha in (source_train_sha256, baseline_sha256)):
        raise ValueError("unknown training recipe or frozen identity")
    if not 80 <= len(train) <= MAX_ORIGINALS or len({row["group_id"] for row in train}) != len(train):
        raise ValueError("training recipe requires bounded independent original training rows")
    if {row["label"] for row in train} != set(RATINGS):
        raise ValueError("training originals must cover every rating label")
    for row in train:
        if (row.get("upstream_license") != "CC-BY-4.0" or row.get("human_review_of_conversion") is not False
                or row.get("annotation_kind") != "constructed-rating-proxy-from-human-attribute"
                or row.get("source_file") != "train.jsonl.gz" or row["wrong_label"] not in RATINGS):
            raise ValueError("training recipe source grant or annotation kind differs")
        context = json.loads(row["prompt"])
        if set(context) != {"task", "output"} or row["chosen"] == row["rejected"]:
            raise ValueError("complete training context and preference required")
    if name == "json-original-v1":
        views, epochs = [dict(row) for row in train], 3
    else:
        views = [evaluation_view(row, fmt, name) for fmt in ("json", "prose") for row in train]
        for label in RATINGS:
            originals = sorted((row for row in train if row["label"] == label), key=lambda row: row["group_id"])
            if len(originals) < BARE_PER_LABEL:
                raise ValueError("insufficient training originals for fixed bare-label allocation")
            views.extend(format_pair(row, "bare_label") for row in originals[:BARE_PER_LABEL])
        if name in {"format-transfer-v2", "format-transfer-v3"}:
            # Fixed source-identity choice, never validation/audit/model selection.
            supplement = dict(min(train, key=lambda row: row["group_id"]))
            for side in ("chosen", "rejected"):
                supplement[side] = json.dumps(json.loads(supplement[side]), ensure_ascii=False, separators=(", ", ": "))
            supplement.update(format="json", chosen_format="json_optimizer_completion", view_role="optimizer_completion")
            views.append(supplement)
        epochs = 1
    stats = view_statistics(views)
    if (Fraction(stats["unique_negative_count"], len(views)) < Fraction(4, 5)
            or Fraction(stats["largest_negative_count"], len(views)) > Fraction(1, 50)):
        raise ValueError("training views exceed unchanged primary repetition limits")
    presentations = len(views) * epochs
    old_presentations = len(train) * 3
    optimizer_ceiling = ((len(views) + GRAD_ACCUM - 1) // GRAD_ACCUM) * epochs
    old_optimizer_ceiling = ((len(train) + GRAD_ACCUM - 1) // GRAD_ACCUM) * 3
    if presentations > old_presentations or optimizer_ceiling > old_optimizer_ceiling:
        raise ValueError("training recipe exceeded the original compute-example budget")
    report = {"schema_version": 1, "recipe": name, "epochs": epochs, "gradient_accumulation": GRAD_ACCUM,
              "original_train_pairs": len(train), "training_view_pairs": len(views),
              "training_source_sha256": source_train_sha256, "baseline_sha256": baseline_sha256,
              "training_views_sha256": digest(pair_bytes(views)), "training_views_bytes": len(pair_bytes(views)),
              "example_presentations_ceiling": presentations, "original_example_presentations_ceiling": old_presentations,
              "optimizer_steps_ceiling": optimizer_ceiling, "original_optimizer_steps_ceiling": old_optimizer_ceiling,
              "bare_per_label_ceiling": BARE_PER_LABEL if name.startswith("format-transfer-") else 0,
              "optimizer_completion_views": 1 if name in {"format-transfer-v2", "format-transfer-v3"} else 0,
              "statistics": stats, "audit_used_for_selection": False, "human_review_of_conversion": False,
              "baseline_fit_scope": "unchanged original JSON training rows, a subset of the derived training views",
              "scope": "constructed format views, not new human annotations or independent source groups"}
    if name == "format-transfer-v3":
        report["presentation_version"] = PRESENTATION_VERSION
    return views, report
