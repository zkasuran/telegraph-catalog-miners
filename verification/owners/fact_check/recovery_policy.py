# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Frozen development ablations after the measured constant-reward collapse."""
import math

STRATEGIES = ("head_only", "low_rate", "head_first")
RECOVERY_RECIPE = {"epochs": 12, "lr": .00001, "weight_decay": .01, "grad_accum": 4,
                   "lora_rank": 8, "lora_alpha": 32, "lora_dropout": 0.,
                   "max_length": 256, "batch_size": 1, "eval_batch": 1, "seed": 42}


def parameter_plan(names, strategy, epoch):
    if strategy not in STRATEGIES or type(epoch) is not int or not 1 <= epoch <= 12:
        raise ValueError("bounded declared recovery strategy and epoch required")
    heads = [name for name in names if "lora_" not in name and ("classifier" in name or "score" in name)]
    loras = [name for name in names if "lora_" in name]
    if not heads or not loras or set(names) != set(heads + loras):
        raise ValueError("exact reward-head and LoRA parameter inventory required")
    head_phase = strategy == "head_only" or strategy == "head_first" and epoch <= 3
    groups = [{"names": heads, "lr": .001 if head_phase else .0005, "weight_decay": 0.}]
    if not head_phase:
        groups.append({"names": loras, "lr": .00001, "weight_decay": .01})
    return groups


def recovery_selection(reports):
    if [item["strategy"] for item in reports] != list(STRATEGIES):
        raise ValueError("all predeclared recovery ablations must complete")
    for item in reports:
        if item["epochs_completed"] != 12 or item["audit_read"] or item["audit_scored"]:
            raise ValueError("complete development-only reports required")
        loss = item["selected_validation"]["val_loss"]
        if type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0:
            raise ValueError("actual finite validation loss required")
    return min(range(len(reports)), key=lambda index: (reports[index]["selected_validation"]["val_loss"], index))
