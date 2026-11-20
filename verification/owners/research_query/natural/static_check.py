# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Exact source-only scope and licence checks, not neural metrics."""
import ast
import json
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    for path in here.glob("*.py"):
        compile(ast.parse(path.read_bytes()), str(path), "exec")
    contract = json.loads((here / "contract.json").read_text())
    if contract["family"] != "RESEARCH_QUERY" or contract["max_tokens"] != 512 or contract["epochs"] != 4 or contract["grad_accum"] != 4:
        raise ValueError("frozen broader research model contract changed")
    if contract["quality"] != {"minimum_accuracy": .8, "minimum_subset_accuracy": .7, "minimum_untrained_gain": .05,
                               "minimum_shortcut_gain": .1, "maximum_shortcut_accuracy": .65}:
        raise ValueError("unchanged quality gates must be retained")
    if contract["audit_used_for_selection"] is not False or contract["scope"] != "Original source human ranks for research quality with broader instruction warm-up; quality annotations affect training confidence, not factual certification":
        raise ValueError("source annotations or audit scope misrepresented")
    if contract.get("model_recipe") != {"lr": .00002, "lora_dropout": .1, "lora_rank": 16, "lora_alpha": 32, "weight_decay": .1}:
        raise ValueError("frozen regularized training improvement changed")
    if contract.get("curriculum") != {"stages": ["warmup", "train", "train", "train"], "warmup_lr": .0001, "confidence_units": [1, 3, 5, 10], "fit_split": "training_only"}:
        raise ValueError("frozen source-confidence curriculum changed")
