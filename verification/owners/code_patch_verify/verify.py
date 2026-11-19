# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""One standalone source-only release gate with actual primary failure controls."""
import json
import sys
import unittest
from pathlib import Path

from ci_reference import score
from data import check_splits, control_metrics, digest
from static_check import main as syntax_and_contracts

HERE = Path(__file__).resolve().parent


def main():
    syntax_and_contracts()
    report = json.loads((HERE / "data/data-report.json").read_text())
    splits = {}
    for name in ("train", "validation", "audit"):
        raw = (HERE / "data" / (name + ".jsonl")).read_bytes()
        if digest(raw) != report["files"][name]["sha256"]:
            raise ValueError("frozen source split bytes changed")
        splits[name] = [json.loads(line) for line in raw.splitlines()]
    checked = check_splits(splits)
    controls = json.loads((HERE / "data/frozen-controls.json").read_text())
    if digest((HERE / "data/frozen-controls.json").read_bytes()) != report["controls_sha256"]:
        raise ValueError("immutable train-fitted controls changed")
    measured = {name: control_metrics(rows, controls) for name, rows in splits.items()}
    if measured != report["control_metrics"] or report["training_allowed"] is not False:
        raise ValueError("fixture scope or frozen controls changed")
    suite = unittest.defaultTestLoader.discover(str(HERE), pattern="test_cpv.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise ValueError("source regression failed")
    for file, expected in (("smoke.json", 1), ("tampered.json", 0)):
        value = json.loads((HERE / "data" / file).read_text())
        actual = score(value["request"], value["answer"], value["evidence"])
        if actual["score"] != expected:
            raise ValueError("real reference primary control failed")
    forbidden = {"torch", "tensorflow", "transformers", "peft", "onnxruntime"}
    if forbidden & {name.partition(".")[0] for name in sys.modules}:
        raise ValueError("source gate imported neural code")
    print(json.dumps({"source_tests": result.testsRun, "normalized_overlap": checked["normalized_overlap"],
                      "shortcuts": measured, "model_operations": "not_attempted", "live_caller_proof": "unverified"}))
    print("ALL GREEN: standalone source/reference only, not a qualified model")


if __name__ == "__main__":
    main()
