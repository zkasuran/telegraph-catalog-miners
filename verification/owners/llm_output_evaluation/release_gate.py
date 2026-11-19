# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Portable source-only gate. Never requires private evidence or model imports."""
import ast
import json
from pathlib import Path
import unittest
from source_gate import HERE, digest
from wire import verify_qualification_render
from data import load_splits


def main():
    for path in HERE.glob("*.py"):
        compile(ast.parse(path.read_bytes()), str(path), "exec")
    for path in HERE.glob("*.json"):
        json.loads(path.read_text())
    contract = verify_qualification_render()
    row = json.loads((HERE / "training-smoke.json").read_text())
    if row["source_file"] != "train.jsonl.gz" or row["human_review_of_conversion"] is not False:
        raise ValueError("primary smoke must be original train provenance, not audit or claimed human review")
    if json.loads(row["chosen"])["rating"] != row["label"]:
        raise ValueError("real source-derived training smoke failed")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(HERE), pattern="test_*.py"))
    if not result.wasSuccessful() or result.skipped:
        raise SystemExit(1)
    from validate_config import validate
    validate()
    print(json.dumps({"source_only": True, "compiled": True, "tests": result.testsRun, "primary_smoke": True,
                      "literal_contract": contract, "model_operations": False}, sort_keys=True))


if __name__ == "__main__":
    main()
