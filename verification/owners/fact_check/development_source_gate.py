# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Actual source-only development smoke. No tokenizer or neural imports."""
import ast
import json
import sys
import tempfile
from pathlib import Path

from data import HERE, digest
from development_policy import RECIPE, prepare_development


def main():
    with tempfile.TemporaryDirectory(dir=HERE) as temporary:
        rows, pins = prepare_development(Path(temporary))
    for filename in ("develop.py", "development_policy.py", "inventory.py"):
        tree = ast.parse((HERE / filename).read_text(), filename=filename)
        compile(tree, filename, "exec")
    workflow = HERE.parents[2] / ".github/workflows/evaluator-owner-fact_check.yml"
    value = json.loads("\n".join(line for line in workflow.read_text().splitlines() if not line.startswith("#")))
    steps = value["jobs"]["factcheck"]["steps"]
    fitting = next(step for step in steps if step.get("id") == "develop")
    if (fitting["if"] != "inputs.mode == 'develop'" or fitting["timeout-minutes"] != 275
            or "chmod 000 verification/owners/fact_check/bundle/audit.jsonl" not in fitting["run"]
            or "test ! -r verification/owners/fact_check/bundle/audit.jsonl" not in fitting["run"]):
        raise ValueError("remote development must deny audit reads before model execution")
    if any(name in sys.modules for name in ("torch", "transformers", "peft", "onnxruntime")):
        raise ValueError("local development source gate imported a neural framework")
    if RECIPE != {"epochs": 24, "lr": .001, "weight_decay": .01, "grad_accum": 4, "lora_rank": 8,
                  "lora_alpha": 32, "lora_dropout": 0., "max_length": 256, "batch_size": 1, "eval_batch": 1, "seed": 42}:
        raise ValueError("development recipe changed without a new explicit contract")
    print(json.dumps({"status": "DEVELOPMENT_SOURCE_PASS", "groups": {name: len(values) for name, values in rows.items()},
                      "split_pins": pins, "baseline_refitted": False, "audit_read": False,
                      "source_manifest_sha256": digest((HERE / "sources.json").read_bytes()), "neural_imports": False}))


if __name__ == "__main__":
    main()
