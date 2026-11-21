# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Training-only convergence comparison after the measured low-rate improvement."""
import math

STRATEGIES = ("low_rate_24", "hard_example_24")
RECIPE = {"epochs": 24, "lr": .00001, "weight_decay": .01, "grad_accum": 4,
          "lora_rank": 8, "lora_alpha": 32, "lora_dropout": 0.,
          "max_length": 256, "batch_size": 1, "eval_batch": 1, "seed": 42}


def parameter_plan(names, strategy, epoch):
    if strategy not in STRATEGIES or type(epoch) is not int or not 1 <= epoch <= 24:
        raise ValueError("declared bounded convergence strategy required")
    heads = [name for name in names if "lora_" not in name and ("classifier" in name or "score" in name)]
    loras = [name for name in names if "lora_" in name]
    if not heads or not loras or set(names) != set(heads + loras):
        raise ValueError("exact reward-head and LoRA inventory required")
    return [{"names": heads, "lr": .0005, "weight_decay": 0.},
            {"names": loras, "lr": .00001, "weight_decay": .01}]


def training_weight(strategy, difference):
    if strategy not in STRATEGIES or type(difference) not in (int, float) or not math.isfinite(difference):
        raise ValueError("actual finite training score difference required")
    return 2. if strategy == "hard_example_24" and difference <= 0 else 1.


def select(reports):
    if [item["strategy"] for item in reports] != list(STRATEGIES):
        raise ValueError("both declared convergence candidates must complete")
    for item in reports:
        if item["epochs_completed"] != 24 or item["audit_read"] or item["audit_scored"]:
            raise ValueError("complete development-only convergence reports required")
        loss = item["selected_validation"]["val_loss"]
        if type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0:
            raise ValueError("finite observed validation loss required")
    return min(range(len(reports)), key=lambda index: (reports[index]["selected_validation"]["val_loss"], index))
