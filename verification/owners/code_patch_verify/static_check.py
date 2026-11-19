# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Standalone source contracts. No neural imports, network or tracked writes."""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
HP = {"max_length": 384, "lora_rank": 8, "lora_alpha": 32, "batch_size": 1, "grad_accum": 16,
      "epochs": 2, "eval_batch": 1, "seed": 42, "early_stopping": False, "grad_checkpointing": False}
QUALITY = {"minimum_accuracy": 0.80, "minimum_subset_accuracy": 0.70,
           "minimum_untrained_gain": 0.05, "maximum_ablated_accuracy": 0.65}


def validate_recipe(recipe):
    if recipe.get("schema_version") != 1 or recipe.get("family") != "CODE_PATCH_VERIFY" or recipe.get("class") != "judgment":
        raise ValueError("wrong family/recipe identity")
    if not re.fullmatch(r"[a-f0-9]{32}", recipe.get("run_id", "")):
        raise ValueError("frozen owner attempt required")
    if recipe.get("max_minutes") != 180 or recipe.get("runner") != "ubuntu-24.04" or recipe.get("repository") != "zkasuran/telegraph-catalog-miners":
        raise ValueError("bounded public CPU recipe required")
    if recipe.get("hyperparameters") != HP or recipe.get("quality") != QUALITY:
        raise ValueError("frozen training or quality gate changed")
    if recipe.get("selection") != {"split": "validation", "metric": "val_loss", "direction": "min", "tie": "earlier_epoch", "audit_after_freeze": True}:
        raise ValueError("only validation can select the frozen candidate")
    if recipe.get("live_evidence_channel_verified") is not False:
        raise ValueError("a recipe flag is not caller proof")


def main():
    paths = sorted(HERE.glob("*.py"))
    for path in paths:
        compile(path.read_bytes(), str(path), "exec")
    validate_recipe(json.loads((HERE / "recipe.json").read_text()))
    pins = json.loads((HERE / "primary-pins.json").read_text())
    for name, expected in pins["files"].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
            raise ValueError("retained primary source/grant changed")
    tree = json.loads((HERE / "evidence/kit_tree.txt").read_text())
    if tree.get("truncated") is not False or len(tree["tree"]) != pins["kit_entries"]:
        raise ValueError("full pinned source inventory required")
    workflow = json.loads((HERE / "workflow.json").read_text())
    if workflow["jobs"]["owner"]["runs-on"] != "ubuntu-24.04" or workflow["jobs"]["owner"]["timeout-minutes"] > 180:
        raise ValueError("workflow CPU/time ceiling changed")
    print(json.dumps({"compiled_python_files": len(paths), "frozen_recipe": "PASS", "pinned_kit_entries": len(tree["tree"])}))


if __name__ == "__main__":
    main()
