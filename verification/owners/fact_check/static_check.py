# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Fixed recipe, rights and workflow source checks. Never imports a model."""
import ast
import json
import re

from data import COUNTS, HERE, digest, pair, source_pool


def main():
    from source_pack import unpack
    unpack()
    contract = json.loads((HERE / "contract.json").read_text())
    if contract["hyperparameters"] != {"max_length": 256, "lora_rank": 8, "lora_alpha": 32,
            "batch_size": 1, "grad_accum": 16, "epochs": 3, "eval_batch": 1, "seed": 42}:
        raise ValueError("fixed three-epoch CPU recipe changed")
    if contract["quality"] != {"minimum_accuracy": .8, "minimum_subset_accuracy": .7,
            "minimum_untrained_gain": .05, "minimum_shortcut_gain": .10, "maximum_shortcut_accuracy": .65}:
        raise ValueError("quality thresholds changed")
    if {name: pin["rows"] for name, pin in contract["splits"].items()} != COUNTS:
        raise ValueError("pilot requires exact 128/32/64 real groups")
    for path in HERE.glob("*.py"):
        ast.parse(path.read_text())
    tree = ast.parse((HERE / "run.py").read_text())
    experiment = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "experiment")
    loop = next(node for node in experiment.body if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "epoch")
    if ast.unparse(loop.iter) != "range(1, EPOCHS + 1)" or any(isinstance(n, (ast.Break, ast.Return)) for n in ast.walk(loop)):
        raise ValueError("all three epochs must complete")
    reads = [n.slice.value for n in ast.walk(loop) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
             and n.value.id == "splits" and isinstance(n.slice, ast.Constant)]
    if "audit" in reads or "validation" not in reads:
        raise ValueError("selection must use development only")
    workflow = HERE.parents[2] / ".github/workflows/evaluator-owner-fact_check.yml"
    value = json.loads("\n".join(line for line in workflow.read_text().splitlines() if not line.startswith("#")))
    job = value["jobs"]["factcheck"]
    if set(value["on"]) != {"workflow_dispatch"} or value["permissions"] != {"contents": "read"}:
        raise ValueError("manually dispatched read-only workflow required")
    if job["runs-on"] != "ubuntu-24.04" or job["timeout-minutes"] != 330:
        raise ValueError("bounded standard public CPU runner required")
    if job.get("if") != "github.repository == 'zkasuran/telegraph-catalog-miners' && github.event.repository.private == false":
        raise ValueError("private-repository compute must be rejected before allocation")
    for step in job["steps"]:
        if "secrets." in str(step) or ("uses" in step and not re.fullmatch(r"[^@]+@[a-f0-9]{40}", step["uses"])):
            raise ValueError("no secret inputs or unpinned actions")
    pins = json.loads((HERE / "sources.json").read_text())
    for name, expected in pins["grants"].items():
        if digest((HERE / name).read_bytes()) != expected:
            raise ValueError("full source grant changed")
    if "All data in Wikidata has a CC0 license." not in (HERE / "source-policy.html").read_text():
        raise ValueError("structured source grant absent")
    from licensing import licence_metadata
    if "x-source-cc0" not in licence_metadata():
        raise ValueError("checkpoint source grant absent")
    for path in [HERE / "README.md", HERE / "NOTICE", *HERE.glob("*.py"), workflow]:
        if re.search("[\u2013\u2014]|, (?:and|or)\\b", path.read_text()):
            raise ValueError("voice gate: " + path.name)
    # Every selected source statement and complete alternatives are independently
    # recomputed from primary bytes. No audit model/shortcut score is computed.
    assertions, _ = source_pool(HERE / "sources")
    bygroup = {a["group_id"]: a for a in assertions}
    capitals = {s["object_id"]: s["object_label"] for assertion in assertions for s in assertion["active_statements"]}
    for name, pin in contract["splits"].items():
        raw = (HERE / "bundle" / (name + ".jsonl")).read_bytes()
        if digest(raw) != pin["sha256"]:
            raise ValueError("frozen split changed")
        for line in raw.decode().split("\n"):
            if line:
                row = json.loads(line)
                if row["source_binding"] != bygroup[row["group_id"]]:
                    raise ValueError("entity/revision/reference/qualifier binding differs")
                candidate = {"object_id": row["claim_object_id"], "object_label": capitals[row["claim_object_id"]]}
                if pair(row["source_binding"], candidate, row["label"] == "True") != row:
                    raise ValueError("model-visible claim or verdict differs from primary bound construction")
    print("FACT_CHECK source bindings, grants, fixed recipe and workflow: PASS")


if __name__ == "__main__":
    main()
