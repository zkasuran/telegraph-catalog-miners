# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Source-only configuration and publication consistency checks."""
import ast
import json
import re
from pathlib import Path


def main():
    import yaml
    here = Path(__file__).resolve().parent
    for path in here.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    contract = json.loads((here / "contract.json").read_text())
    hp = contract["hyperparameters"]
    if hp != {"max_length": 256, "lora_rank": 8, "lora_alpha": 32, "batch_size": 1, "grad_accum": 16, "epochs": 3, "eval_batch": 1, "seed": 42}:
        raise ValueError("fixed CPU recipe contract changed")
    if contract["quality"] != {"minimum_accuracy": .80, "minimum_subset_accuracy": .70, "minimum_untrained_gain": .05,
                               "minimum_shortcut_gain": .10, "maximum_shortcut_accuracy": .65}:
        raise ValueError("fixed quality contract changed")
    workflow = here.parents[1] / ".github/workflows/evaluator-qa-train.yml"
    value = yaml.load(workflow.read_text(), Loader=yaml.BaseLoader)
    job = value["jobs"]["qa"]
    if value["on"] != {"workflow_dispatch": ""} or value["permissions"] != {"contents": "read"}:
        raise ValueError("workflow trigger or permissions changed")
    if job["runs-on"] != "ubuntu-24.04" or job["timeout-minutes"] != "330":
        raise ValueError("public CPU or total ceiling changed")
    if len(job["steps"]) != 8 or any("secrets." in str(step) for step in job["steps"]):
        raise ValueError("workflow step or credential contract changed")
    for step in job["steps"]:
        if "uses" in step and not re.fullmatch(r"[^@]+@[a-f0-9]{40}", step["uses"]):
            raise ValueError("action must use an exact source SHA")
    checkout = job["steps"][0]["with"]
    if "/.github/workflows/evaluator-qa-train.yml" not in checkout["sparse-checkout"]:
        raise ValueError("sparse checkout must retain the workflow used by the static gate")
    for path in [here / "README.md", here / "NOTICE", *here.glob("*.py"), *here.glob("*.go"), workflow]:
        if re.search("[\u2013\u2014]|, (?:and|or)\\b", path.read_text()):
            raise ValueError("outward voice gate: " + path.name)
    from licensing import OWN_ID, OWN_SHA, licence_metadata
    if contract["selection"] != {"split": "validation", "metric": "val_loss", "direction": "min", "tie": "earlier_epoch",
                                 "epochs_completed": 3, "early_stopping": False, "audit_after_selection": True,
                                 "evaluation_rng": "fork CPU Torch and restore Python/NumPy", "training_sampler_rng": "dedicated CPU generator seed 42"}:
        raise ValueError("development-only selection contract changed")
    for path in [*here.glob("*.py"), *here.glob("*.go"), here / "verify.sh", workflow]:
        if "SPDX-License-Identifier: " + OWN_ID not in path.read_text():
            raise ValueError("own code/workflow lacks its SAND header: " + path.name)
    if __import__("hashlib").sha256((here / "LICENSE").read_bytes()).hexdigest() != OWN_SHA:
        raise ValueError("own SAND terms differ from reviewed catalog licence")
    licence_metadata()
    training = ast.parse((here / "run.py").read_text())
    experiment = next(node for node in training.body if isinstance(node, ast.FunctionDef) and node.name == "experiment")
    loop = next(node for node in experiment.body if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "epoch")
    if ast.unparse(loop.iter) != "range(1, EPOCHS + 1)":
        raise ValueError("training does not complete all fixed epochs")
    if any(isinstance(node, (ast.Break, ast.Return)) for node in ast.walk(loop)):
        raise ValueError("training has an early-stop path")
    accesses = [node for node in ast.walk(loop) if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == "splits" and isinstance(node.slice, ast.Constant)]
    if any(node.slice.value == "audit" for node in accesses) or not any(node.slice.value == "validation" for node in accesses):
        raise ValueError("epoch selection must use development and never audit")
    raw = (here / "UPSTREAM-APACHE-2.0.txt").read_bytes()
    pins = json.loads((here / "sources.json").read_text())
    if __import__("hashlib").sha256(raw).hexdigest() != pins["files"]["oasst-license.txt"][1]:
        raise ValueError("full Apache source terms mismatch")
    if __import__("hashlib").sha256((here / "sources.json").read_bytes()).hexdigest() != contract["freeze_sha256"]["sources.json"]:
        raise ValueError("source pin manifest changed")
    print("QA AST, workflow YAML, three epochs, development selection, own SAND and unchanged upstream terms: PASS")


if __name__ == "__main__":
    main()
