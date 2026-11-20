# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Standalone source-only contracts and original preference smoke."""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

from data import HERE, digest, read_splits
from static_check import main as static_check


def main():
    static_check()
    report, splits = read_splits(HERE / "fixtures")
    gate = json.loads((HERE / "fixtures/dataset-gate.json").read_text())
    if gate["ok"] is not True or gate["audit_scored"] is not False or max(gate["validation_controls"].values()) > .65:
        raise ValueError("original dataset or immutable validation shortcut gate failed")
    contract = json.loads((HERE / "contract.json").read_text())
    for name, expected in contract["source_hashes"].items():
        if digest((HERE / name).read_bytes()) != expected:
            raise ValueError("natural source bytes changed: " + name)
    for name, expected in contract["fixture_hashes"].items():
        if digest((HERE / "fixtures" / name).read_bytes()) != expected:
            raise ValueError("frozen original corpus changed")
    if len(splits["train"]) != 768 or len({r["group_id"] for r in splits["train"]}) != 768 or len(splits["audit"]) != 64 or report["official_validation_quarantined"] is not True:
        raise ValueError("complete independent source cohorts required")
    source = ast.parse((HERE / "run_base.py").read_text())
    experiment = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "experiment")
    loop = next(node for node in experiment.body if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "epoch")
    if ast.unparse(loop.iter) != "range(1, EPOCHS + 1)" or any(isinstance(node, (ast.Break, ast.Return)) for node in ast.walk(loop)):
        raise ValueError("frozen complete epoch selection changed")
    accesses = [node.slice.value for node in ast.walk(loop) if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "splits" and isinstance(node.slice, ast.Constant)]
    if "audit" in accesses or "validation" not in accesses:
        raise ValueError("audit entered epoch selection")
    command = [sys.executable, "-W", "error", "-m", "unittest", "discover", "-s", str(HERE), "-p", "test_natural.py", "-v"]
    result = subprocess.run(command, capture_output=True, timeout=120, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    log = result.stdout + result.stderr
    (HERE / "evidence").mkdir(exist_ok=True)
    (HERE / "evidence/source-tests.log").write_bytes(log)
    sys.stdout.buffer.write(log)
    if result.returncode or "torch" in sys.modules:
        raise ValueError("natural source smoke/failure controls failed or neural module imported")
    print("ALL NATURAL SOURCE CHECKS GREEN")


if __name__ == "__main__":
    main()
