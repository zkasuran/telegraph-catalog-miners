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


def main():
    for path in HERE.glob("*.py"):
        ast.parse(path.read_text(), filename=path.name)
        if "SPDX-License-Identifier: " + OWN_ID not in path.read_text():
            raise ValueError("own source lacks its licence: " + path.name)
    licence_metadata()
    contract = json.loads((HERE / "contract.json").read_text())
    if contract["family"] != FAMILY or contract["child"] != FAMILY + "_GENERAL":
        raise ValueError("exact owner family required")
    if contract["hyperparameters"] != {"max_length": 256, "lora_rank": 8, "lora_alpha": 32, "batch_size": 1, "grad_accum": 16, "epochs": 3, "eval_batch": 1, "seed": 42}:
        raise ValueError("fixed CPU recipe changed")
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
    if set(inputs) != {"mode", "owner_attempt", "serving_settings"} or inputs["mode"]["options"] != ["train", "serve"]:
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
    training = ast.parse((HERE / "remote.py").read_text())
    experiment = next(n for n in training.body if isinstance(n, ast.FunctionDef) and n.name == "experiment")
    loop = next(n for n in experiment.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "epoch")
    if ast.unparse(loop.iter) != "range(1, EPOCHS + 1)" or any(isinstance(n, (ast.Break, ast.Return)) for n in ast.walk(loop)):
        raise ValueError("all three fixed epochs must complete")
    accesses = [n.slice.value for n in ast.walk(loop) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "splits" and isinstance(n.slice, ast.Constant)]
    if "validation" not in accesses or "audit" in accesses:
        raise ValueError("audit must remain outside development selection")
    for source in ("remote.py", "runtime_check.py", "serve.py"):
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
