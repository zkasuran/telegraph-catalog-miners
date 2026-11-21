# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Actual source compilation and closed schema checks without neural imports."""
import ast
import json
import sys

from contracts import HERE, WORKFLOW, sha, validate_experiment


def main():
    experiment = json.loads((HERE / "experiment.json").read_bytes())
    validate_experiment(experiment)
    from recover import validate_plan
    validate_plan(json.loads((HERE / "recovery.json").read_bytes()))
    from balanced import plan as balanced_plan
    balanced = balanced_plan()
    from expanded import plan as expanded_plan
    expanded = expanded_plan()
    if sha((HERE / "expanded-exclusions.json").read_bytes()) != expanded["exclusions_sha256"] or len(json.loads((HERE / "expanded-exclusions.json").read_bytes())) != 512:
        raise ValueError("both exposed audit inventories must be fully excluded")
    for field, hash_field in (("checkpoint", "checkpoint_sha256"), ("config", "config_sha256")):
        if sha((HERE / "assets" / expanded["warm_start"][field]).read_bytes()) != expanded["warm_start"][hash_field]:
            raise ValueError("licensed M2 initialization changed")
    if sha((HERE / "excluded-audit-groups.json").read_bytes()) != balanced["excluded_audit_groups_sha256"]:
        raise ValueError("complete exposed-audit exclusion file changed")
    for field, hash_field in (("checkpoint", "checkpoint_sha256"), ("config", "config_sha256")):
        if sha((HERE / "assets" / balanced["warm_start"][field]).read_bytes()) != balanced["warm_start"][hash_field]:
            raise ValueError("licensed B3 warm start changed")
    warm = experiment["warm_start"]
    for name, expected in ((warm["checkpoint"], warm["checkpoint_sha256"]), (warm["config"], warm["config_sha256"]),
                           (warm["untrained"], warm["untrained_sha256"]), (warm["untrained_config"], warm["config_sha256"]),
                           (warm["coefficients"], warm["coefficients_sha256"])):
        if sha((HERE / "assets" / name).read_bytes()) != expected:
            raise ValueError("fitted/untrained asset or frozen coefficient bytes changed")
    support = json.loads((HERE / "SUPPORT.json").read_bytes())
    for row in support["files"]:
        path = HERE / row["path"]
        if not path.is_file() or sha(path.read_bytes()) != row["sha256"]:
            raise ValueError("packaged support source/hash changed")
    count = 0
    for path in HERE.rglob("*.py"):
        compile(path.read_text(), str(path), "exec")
        count += 1
        if path.parent == HERE:
            tree = ast.parse(path.read_text())
            used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                    if any(name.split(".")[0] in {"torch", "numpy", "peft", "transformers", "tokenizers"} for name in names):
                        raise ValueError("neural import at local source-contract scope")
                    bindings = [item.asname or (item.name.split(".")[0] if isinstance(node, ast.Import) else item.name) for item in node.names]
                    if any(name not in used for name in bindings):
                        raise ValueError("unused owned source import: " + path.name)
    root = HERE.parents[2]
    workflow = json.loads((root / ".github/workflows" / WORKFLOW).read_bytes())
    if workflow["permissions"] != {"contents": "read"} or set(workflow["jobs"]) != {"owner", "aegis_fit", "aegis_audit"}:
        raise ValueError("workflow privilege/concurrency changed")
    job = workflow["jobs"]["owner"]
    if job["runs-on"] != "ubuntu-24.04" or job["timeout-minutes"] != 330:
        raise ValueError("workflow is not the bounded free CPU contract")
    if workflow["concurrency"]["cancel-in-progress"] is not False:
        raise ValueError("workflow may cancel a preserved job")
    mode = workflow["on"]["workflow_dispatch"]["inputs"]["mode"]
    if mode["options"] != ["expanded", "serve", "aegis"] or mode["default"] != "expanded":
        raise ValueError("new training requires the clean independent audit pipeline")
    if workflow["jobs"]["aegis_audit"].get("needs") != "aegis_fit":
        raise ValueError("audit must consume frozen validation-selected fit, never fit audit")
    for policy_job in (workflow["jobs"]["aegis_fit"], workflow["jobs"]["aegis_audit"]):
        if policy_job["runs-on"] != "ubuntu-24.04" or policy_job["timeout-minutes"] != 330 or "aegis" not in policy_job["if"]:
            raise ValueError("A1 must remain mode-fenced free bounded public CPU")
    expanded_steps = [step for step in job["steps"] if "expanded_remote.py run" in step.get("run", "")]
    if len(expanded_steps) != 1 or expanded_steps[0]["timeout-minutes"] != 295 or any("remote.py run" in step.get("run", "") and "expanded_remote.py run" not in step.get("run", "") for step in job["steps"]):
        raise ValueError("failed audit cannot be rerun as a new training result")
    result_steps = [step for step in job["steps"] if step.get("with", {}).get("name", "").startswith("owner-registration-")]
    if len(result_steps) != 1 or "success()" not in result_steps[0].get("if", "") or "registration-ready.json" not in result_steps[0]["with"]["path"] or "/evidence/\n" in result_steps[0]["with"]["path"]:
        raise ValueError("bounded registration evidence must be separate from epoch diagnostics")
    if any(not path.startswith("${{ runner.temp }}/cm-training/evidence/") for path in result_steps[0]["with"]["path"].splitlines()):
        raise ValueError("actual adapter needs registration-ready.json at the ZIP root, not a nested common ancestor")
    for name in ("LICENSE", "NOTICE", "DATA-SOURCES.md", "SECURITY.md"):
        if not (HERE / name).is_file() or (HERE / name).stat().st_size < 100:
            raise ValueError("missing publication rights/security file")
    if "Source-Available No-Derivatives" not in (HERE / "LICENSE").read_text():
        raise ValueError("competitive licence missing")
    if any(name in sys.modules for name in ("torch", "transformers", "tokenizers", "peft", "numpy")):
        raise ValueError("local gate loaded a model framework")
    print("SOURCE GATE PASS: compiled " + str(count) + " Python files, validated support hashes, rights files and exact CPU workflow")


if __name__ == "__main__":
    main()
