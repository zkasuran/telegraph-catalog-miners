# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""The primary kit's TrainingRecord shape, without neural imports."""
import math

from contracts import FAMILY, FORMATS, VALIDATION_SHA


def validate_record(value):
    if not isinstance(value, dict) or value.get("family") != FAMILY or not isinstance(value.get("children"), list) or not value["children"]:
        raise ValueError("training record: children is empty or family differs")
    seen = set()
    for child in value["children"]:
        name = child.get("intent", "")
        if not isinstance(name, str) or not (name == FAMILY or name.startswith(FAMILY + "_")) or name in seen:
            raise ValueError("training record: invalid or duplicate child intent")
        seen.add(name)
        if type(child.get("pairs")) is not int or child["pairs"] < 0:
            raise ValueError("training record: pair count is invalid")
        accuracy = child.get("accuracy")
        if type(accuracy) not in (int, float) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("training record: accuracy is invalid")
        if type(child.get("held_out")) is not bool:
            raise ValueError("training record: held_out must be explicit")
        if "spearman" in child and (type(child["spearman"]) not in (int, float) or not math.isfinite(child["spearman"]) or not -1 <= child["spearman"] <= 1):
            raise ValueError("training record: spearman is invalid")


def development_record(metrics, pairs=64):
    if set(metrics) != FORMATS:
        raise ValueError("actual validation metrics for every format required")
    accuracy = sum(metrics[fmt]["accuracy"] for fmt in FORMATS) / len(FORMATS)
    record = {"family": FAMILY, "children": [{"intent": FAMILY + "_GENERAL", "dataset_version": VALIDATION_SHA,
                                               "pairs": pairs, "accuracy": accuracy, "held_out": False}],
              "trainer": "pinned ModernBERT LoRA Bradley-Terry", "backbone": "ModernBERT-base"}
    validate_record(record)
    return record
