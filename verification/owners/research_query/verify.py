# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Portable source-only packaged release gate. Never imports neural runtimes."""
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from prepare import check_splits
from source import read_url, sha, write_json

HERE = Path(__file__).resolve().parent


def verify():
    contract = json.loads((HERE / "contract.json").read_text())
    if contract["schema_version"] != 1 or contract["family"] != "RESEARCH_QUERY" or contract["child"] != "RESEARCH_QUERY_GENERAL" or contract["supported_formats"] != ["prose", "json"]:
        raise ValueError("exact family and supported-format contract required")
    for name, expected in contract["owned_source_sha256"].items():
        if sha((HERE / name).read_bytes()) != expected:
            raise ValueError("owned public source changed after freeze: " + name)
    for name, expected in contract["shared_source_sha256"].items():
        if name in ("packagecheck.go", "packagecheck_test.go", "licensing.py"):
            name = "qa-experiment/" + name
        if sha((HERE / "support" / name).read_bytes()) != expected:
            raise ValueError("retained upstream source changed: " + name)
    for name, expected in contract["data_artifacts_sha256"].items():
        if sha((HERE / "data" / name).read_bytes()) != expected:
            raise ValueError("immutable source artifact changed: " + name)
    for path in HERE.glob("*.py"):
        tree = ast.parse(path.read_bytes(), filename=str(path))
        compile(tree, str(path), "exec")
        loaded = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        imported = [a.asname or a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names]
        if any(name not in loaded for name in imported):
            raise ValueError("unused-import lint: " + path.name)
    remote = ast.parse((HERE / "remote.py").read_bytes())
    for node in remote.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("main", "experiment", "runtime_check", "native_scores", "record_audit", "record_untrained"):
            if ast.unparse(node.body[0]) != "remote_only()":
                raise ValueError("neural operation missing first-statement runner guard")
    core = ast.parse((HERE / "train_core.py").read_bytes())
    experiment = next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == "experiment")
    epoch = next(n for n in experiment.body if isinstance(n, ast.For) and ast.unparse(n.target) == "epoch")
    if ast.unparse(epoch.iter) != "range(1, 4)" or any(isinstance(n, (ast.Break, ast.Return)) for n in ast.walk(epoch)):
        raise ValueError("fixed training epochs changed")
    accesses = [n.slice.value for n in ast.walk(epoch) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "splits" and isinstance(n.slice, ast.Constant)]
    if "audit" in accesses or "validation" not in accesses:
        raise ValueError("audit used for checkpoint selection")
    splits = {}
    for name, pin in contract["splits"].items():
        raw = (HERE / "data" / (name + ".jsonl")).read_bytes()
        if sha(raw) != pin["sha256"]:
            raise ValueError("original frozen split bytes changed")
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != pin["rows"] or len({r["group_id"] for r in rows}) != pin["groups"] or any(r["human_reviewed"] is not False for r in rows):
            raise ValueError("original grouped counts or label provenance changed")
        splits[name] = rows
    check_splits(splits)
    source_manifest = (HERE / "data/source-manifest.json").read_bytes()
    if sha(source_manifest) != contract["source_manifest_sha256"]:
        raise ValueError("complete source manifest binding changed")
    row = splits["train"][0]
    path = HERE / "evidence/xml" / row["source_path"]
    if not path.exists():
        raw, receipt = read_url(row["source_url"], 4 << 20)
        if sha(raw) != row["source_sha256"]:
            raise ValueError("real original source smoke hash changed")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        write_json(HERE / "evidence/smoke-receipt.json", receipt)
    import yaml
    workflow = yaml.load((HERE / "workflow.yml").read_text(), Loader=yaml.BaseLoader)
    job = workflow["jobs"]["research"]
    if job["runs-on"] != "ubuntu-24.04" or job["timeout-minutes"] != "330" or workflow["permissions"] != {"contents": "read"} or workflow["concurrency"]["cancel-in-progress"] != "false":
        raise ValueError("CPU, timeout or authority gate failed")
    if workflow["on"]["workflow_dispatch"]["inputs"]["mode"]["options"] != ["train", "serve"]:
        raise ValueError("only train/serve modes are allowed")
    for step in job["steps"]:
        if "uses" in step and not re.fullmatch(r"actions/[a-z-]+@[a-f0-9]{40}", step["uses"]):
            raise ValueError("exact action commit required")
        if "secrets." in str(step):
            raise ValueError("no provider or signer credentials in public jobs")
    command = [sys.executable, "-W", "error", "-m", "unittest", "discover", "-s", str(HERE), "-p", "test_*.py", "-v"]
    result = subprocess.run(command, capture_output=True, timeout=120, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    log = result.stdout + result.stderr
    (HERE / "evidence/local-tests.log").write_bytes(log)
    sys.stdout.buffer.write(log)
    if result.returncode or "torch" in sys.modules:
        raise ValueError("source/contract tests failed or neural runtime imported")
    write_json(HERE / "evidence/local-verification.json", {"family": "RESEARCH_QUERY", "contract_sha256": sha((HERE / "contract.json").read_bytes()),
               "syntax_and_ast_lint": True, "source_identity": True, "workflow_parsed_and_bounded": True,
               "neural_imports": False, "audit_quality_measured": False, "test_log_sha256": sha(log)})
    print("ALL SOURCE CHECKS GREEN")


if __name__ == "__main__":
    verify()
