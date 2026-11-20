# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Parse and validate the actual standalone workflow and fixed source contracts."""
import ast
import json
import re
import subprocess
import sys

from data import FAMILY, HERE, digest
from licensing import OWN_ID, licence_metadata

WORKFLOW = HERE.parents[2] / ".github/workflows/evaluator-owner-text_summarization.yml"


def check_curriculum_training(source):
    from role_bootstrap import WARMUP_EPOCHS, WARMUP_LR, WARMUP_STEPS
    if (WARMUP_EPOCHS, WARMUP_LR, WARMUP_STEPS) != (64, .0001, 256):
        raise ValueError("fixed train-only warm-up constants changed")
    tree = ast.parse(source)
    imports = {n.module: {a.name for a in n.names} for n in tree.body if isinstance(n, ast.ImportFrom)}
    if imports.get("role_bootstrap") != {"WARMUP_EPOCHS", "WARMUP_LR", "WARMUP_STEPS", "admit", "warmup_plan", "warmup_rows"}:
        raise ValueError("actual runtime must import the frozen role-bootstrap implementation")
    experiment = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "experiment")
    assignments = {t.id: n for n in experiment.body if isinstance(n, ast.Assign)
                   for t in n.targets if isinstance(t, ast.Name)}
    required = {
        "warm": "warmup_rows(manifest_source, splits['train'])",
        "warm_schedule": "warmup_plan(warm)",
        "warm_optimizer": "torch.optim.AdamW(params, lr=WARMUP_LR, weight_decay=hp.weight_decay)",
        "optimizer": "torch.optim.AdamW(params, lr=hp.lr, weight_decay=hp.weight_decay)",
    }
    for name, expression in required.items():
        if name not in assignments or ast.unparse(assignments[name].value) != expression:
            raise ValueError("train-only curriculum or fresh main optimizer contract changed")
    warm_loop = next(n for n in experiment.body if isinstance(n, ast.For) and ast.unparse(n.iter) == "range(WARMUP_EPOCHS)")
    main_loop = next(n for n in experiment.body if isinstance(n, ast.For) and ast.unparse(n.iter) == "range(1, EPOCHS + 1)")
    if not assignments["warm_optimizer"].lineno < warm_loop.lineno < assignments["optimizer"].lineno < main_loop.lineno:
        raise ValueError("complete warm-up must precede fresh-optimizer main training")
    if any(isinstance(n, (ast.Break, ast.Return)) for n in ast.walk(warm_loop)):
        raise ValueError("warm-up may not stop early")
    if "warm_steps != WARMUP_STEPS" not in source or 'save_checkpoint(model, root / "warmup")' not in source:
        raise ValueError("complete warm-up count and preserved checkpoint required")
    if "warm_loss.backward()" not in source or "if not admit(bootstrap_metrics):" not in source:
        raise ValueError("actual batch-size-2 bootstrap loss and training-fit admission required")


def main():
    from training_policy import EPOCHS
    if EPOCHS != 9:
        raise ValueError("fixed nine main epochs required in the executed policy")
    for path in HERE.glob("*.py"):
        ast.parse(path.read_text(), filename=path.name)
        if "SPDX-License-Identifier: " + OWN_ID not in path.read_text():
            raise ValueError("own source lacks its licence: " + path.name)
    licence_metadata()
    contract = json.loads((HERE / "contract.json").read_text())
    if contract["family"] != FAMILY or contract["child"] != FAMILY + "_GENERAL":
        raise ValueError("exact owner family required")
    if contract["hyperparameters"] != {"max_length": 256, "lora_rank": 8, "lora_alpha": 32, "batch_size": 1, "grad_accum": 4, "epochs": 9, "eval_batch": 1, "seed": 42, "lr": .0001, "lora_dropout": 0.0}:
        raise ValueError("fixed CPU recipe changed")
    if contract.get("recipe_version") != 4 or contract.get("curriculum") != {"split": "train", "view": "original_complete_two_reference", "unique_groups": 4, "unique_pairs": 8, "epochs": 64, "batch_size": 2, "learning_rate": .0001, "optimizer_steps": 256, "minimum_training_accuracy": 1.0}:
        raise ValueError("explicit source/training revision required")
    if contract["quality"] != {"minimum_accuracy": .80, "minimum_subset_accuracy": .70, "minimum_untrained_gain": .05, "minimum_shortcut_gain": .10, "maximum_shortcut_accuracy": .65}:
        raise ValueError("unchanged quality thresholds required")
    if digest((HERE / "data/manifest.json").read_bytes()) != contract["source_manifest_sha256"]:
        raise ValueError("source manifest identity changed")
    if digest((HERE / "data/frozen-count-baselines.json").read_bytes()) != contract["baseline_sha256"]:
        raise ValueError("immutable baseline changed")
    workflow_raw = WORKFLOW.read_text()
    if not workflow_raw.startswith("# SPDX-License-Identifier: " + OWN_ID + "\n"):
        raise ValueError("workflow licence header required")
    workflow = json.loads(workflow_raw.split("\n", 1)[1])
    if set(workflow) != {"name", "on", "permissions", "concurrency", "jobs"} or workflow["permissions"] != {"contents": "read"}:
        raise ValueError("closed read-only workflow schema required")
    if set(workflow["on"]) != {"workflow_dispatch"} or workflow["concurrency"]["cancel-in-progress"] is not False:
        raise ValueError("explicit dispatch with no cancellation required")
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    if set(inputs) != {"mode", "owner_attempt", "serving_settings"} or inputs["mode"]["options"] != ["train", "serve", "diagnostic", "diagnostic_reciprocal"]:
        raise ValueError("closed training and serving modes required")
    if set(workflow["jobs"]) != {"owner"}:
        raise ValueError("one bounded public CPU job required")
    job = workflow["jobs"]["owner"]
    if job["runs-on"] != "ubuntu-24.04" or job["timeout-minutes"] != 350 or "strategy" in job:
        raise ValueError("public CPU and bounded timeout required")
    for step in job["steps"]:
        if "secrets." in str(step) or step.get("timeout-minutes", 0) > 350:
            raise ValueError("workflow credential or timeout boundary failed")
        if "uses" in step and not re.fullmatch(r"[^@]+@[a-f0-9]{40}", step["uses"]):
            raise ValueError("exact action source revision required")
        if "run" in step:
            subprocess.run(["bash", "-n", "-c", step["run"]], check=True, timeout=5)
        if step.get("name") == "Upload ONLY restricted public lease" and step["with"]["path"] != "${{ runner.temp }}/summary-serving/lease.json":
            raise ValueError("serving may upload only its restricted public lease")
        if step.get("name") == "Upload bounded registration-ready report only" and step["with"]["path"] != "${{ runner.temp }}/summary-experiment/registration-ready.json":
            raise ValueError("registration evidence must be a bounded root report, not the checkpoint archive")
    training = ast.parse((HERE / "remote.py").read_text())
    check_curriculum_training((HERE / "remote.py").read_text())
    experiment = next(n for n in training.body if isinstance(n, ast.FunctionDef) and n.name == "experiment")
    loop = next(n for n in experiment.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "epoch")
    if ast.unparse(loop.iter) != "range(1, EPOCHS + 1)" or any(isinstance(n, (ast.Break, ast.Return)) for n in ast.walk(loop)):
        raise ValueError("all nine fixed main epochs must complete")
    accesses = [n.slice.value for n in ast.walk(loop) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "splits" and isinstance(n.slice, ast.Constant)]
    if "validation" not in accesses or "audit" in accesses:
        raise ValueError("audit must remain outside development selection")
    positions = {name: node.lineno for node in ast.walk(experiment) if isinstance(node, ast.Assign)
                 for target in node.targets if isinstance(target, ast.Name)
                 for name in (target.id,) if name in {"admission", "audit_views"}}
    guarded = any(isinstance(node, ast.If) and ast.unparse(node.test) == "not admission['audit_allowed']"
                  and any(isinstance(child, ast.Return) for child in ast.walk(node)) for node in ast.walk(experiment))
    if not guarded or positions.get("admission", 1 << 30) >= positions.get("audit_views", 0):
        raise ValueError("development admission must precede any neural audit scoring")
    for source in ("remote.py", "runtime_check.py", "serve.py", "diagnostic.py", "reciprocal.py"):
        tree = ast.parse((HERE / source).read_text())
        for node in tree.body:
            if isinstance(node, ast.Import) and any(n.name.split(".")[0] in {"torch", "peft", "transformers", "onnxruntime"} for n in node.names):
                raise ValueError("neural import outside remote entry point")
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in {"torch", "peft", "transformers", "onnxruntime"}:
                raise ValueError("neural import outside remote entry point")
    for name in ("README.md", "NOTICE", "SECURITY.md"):
        if re.search("[\u2013\u2014]|, (?:and|or)\\b", (HERE / name).read_text()):
            raise ValueError("outward prose voice check failed: " + name)
    if any(name in sys.modules for name in ("torch", "peft", "transformers", "onnxruntime")):
        raise ValueError("source gate imported a neural runtime")
    print("Source AST, actual workflow, frozen recipe, selection and grants: PASS")


if __name__ == "__main__":
    main()
