# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Run only data and shortcut checks. Never import or load a model."""
import argparse
import json
import sys
from pathlib import Path

from data import write_json
from run import read_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.kit.resolve()))
    from float_runtime.train_adapter import Hyperparams, gate_dataset
    _, splits = read_splits(args.data)
    report = gate_dataset(splits["train"], Hyperparams(max_length=256))
    if "torch" in sys.modules:
        raise RuntimeError("data-only check imported torch")
    write_json(args.data / "dataset-gate.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
