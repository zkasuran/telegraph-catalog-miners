# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Standalone stdlib release gate. No network, model imports or file mutation."""
import ast
import json
import re
import sys
from pathlib import Path

from contracts import source_gate
from licensing import licence_metadata

HERE = Path(__file__).resolve().parent


def main():
    for path in HERE.glob("*.py"):
        source = path.read_text()
        ast.parse(source, filename=path.name)
        if "SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0" not in source:
            raise ValueError("source terms absent")
    workflow = json.loads((HERE / "workflow.json").read_text())["workflow"]
    public_workflow = HERE.parents[2] / ".github/workflows/evaluator-owner-web_search.yml"
    if public_workflow.is_file() and json.loads(public_workflow.read_text().partition("\n")[2]) != workflow:
        raise ValueError("actual owned workflow differs from validated schema")
    if workflow["on"] != {"workflow_dispatch": {"inputs": workflow["on"]["workflow_dispatch"]["inputs"]}} or workflow["permissions"] != {"contents": "read"}:
        raise ValueError("manual bounded workflow required")
    if len(workflow["jobs"]) != 1:
        raise ValueError("one bounded CPU job required")
    job = workflow["jobs"]["owner"]
    if job["runs-on"] != "ubuntu-24.04" or not 1 <= job["timeout-minutes"] <= 350:
        raise ValueError("public CPU and explicit time ceiling")
    for step in job["steps"]:
        if "uses" in step and not re.fullmatch(r"[^@]+@[a-f0-9]{40}", step["uses"]):
            raise ValueError("exact action source SHA required")
    for name in ("README.md", "DATA-SOURCES.md", "SECURITY.md", "NOTICE"):
        if re.search("[\u2013\u2014]|, (?:and|or)\\b", (HERE / name).read_text()):
            raise ValueError("outward text failed")
    licence_metadata()
    print(json.dumps(source_gate()))
    if {"torch", "transformers", "peft", "onnxruntime"} & set(sys.modules):
        raise RuntimeError("source gate imported a model library")


if __name__ == "__main__":
    main()
