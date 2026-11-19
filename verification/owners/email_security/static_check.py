# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Compile source and validate exact bounded workflow/config without models."""
import ast
import argparse
import json
from pathlib import Path

from data import digest, read_splits
from remote import validate_recipe

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    args = parser.parse_args()
    import yaml
    recipe = json.loads((HERE / "recipe.json").read_bytes())
    validate_recipe(recipe)
    workflow = yaml.load((HERE / "workflow.yml").read_text(), Loader=yaml.BaseLoader)
    if (workflow["name"] != "evaluator-owner-email_security" or set(workflow["on"]) != {"workflow_dispatch"}
            or workflow["permissions"] != {"contents": "read"} or set(workflow["jobs"]) != {"train", "serve"}
            or workflow["concurrency"]["cancel-in-progress"] != "false"):
        raise ValueError("workflow authority or concurrency changed")
    if workflow["on"]["workflow_dispatch"]["inputs"]["mode"]["options"] != ["train", "serve"]:
        raise ValueError("unsupported workflow mode")
    for name, job in workflow["jobs"].items():
        if job["runs-on"] != "ubuntu-24.04" or job["timeout-minutes"] != {"train": "150", "serve": "350"}[name] or "strategy" in job:
            raise ValueError("workflow CPU/time ceiling")
        for step in job["steps"]:
            if "uses" in step and not __import__("re").fullmatch(r"actions/[a-z-]+@[0-9a-f]{40}", step["uses"]):
                raise ValueError("unpinned action")
    raw = (HERE / "workflow.yml").read_text()
    if "secrets." in raw or "ssh " in raw or "persist-credentials: false" not in raw:
        raise ValueError("workflow secret or account surface")
    compiled = []
    for path in sorted(HERE.glob("*.py")):
        source = path.read_bytes()
        compile(ast.parse(source, filename=str(path)), str(path), "exec")
        compiled.append(path.name)
    report, _, frozen = read_splits(args.data)
    contract = json.loads((HERE / "contract.json").read_bytes())
    if (contract["family"] != "EMAIL_SECURITY" or contract["files"] != report["files"]
            or contract["frozen_sha256"] != report["freeze_sha256"] or frozen["fit_split"] != "train"):
        raise ValueError("immutable data contract changed")
    if digest((HERE / "frozen-controls.json").read_bytes()) != contract["frozen_sha256"]:
        raise ValueError("immutable published baseline changed")
    for name, sha in contract["source_sha256"].items():
        if digest((HERE / name).read_bytes()) != sha:
            raise ValueError("frozen source changed: " + name)
    print("PASS syntax", len(compiled), "Python files; workflow and recipe validated; immutable split/source hashes match")


if __name__ == "__main__":
    main()
