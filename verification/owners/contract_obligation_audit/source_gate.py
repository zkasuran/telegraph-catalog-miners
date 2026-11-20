# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Refuse source constructions solved by a frozen shortcut, before neural work."""
import argparse
import json
from pathlib import Path

from baselines import measure, validate_frozen
from contracts import HERE, FORMATS, check_config, read_jsonl, strict_json, write_json
from data import check_splits, pair


def verdict(splits, frozen, config):
    check_splits(splits, config)
    validate_frozen(frozen, splits["train"])
    checks = {split: {style: measure([pair(row, style) for row in splits[split]], frozen)
                      for style in FORMATS} for split in ("train", "validation")}
    failures = []
    for split, formats in checks.items():
        for style, shortcuts in formats.items():
            strongest = max(shortcuts.values())
            if strongest > .65:
                failures.append({"split": split, "format": style, "strongest_shortcut": strongest,
                                 "reason": "frozen source-only shortcut exceeds the unchanged 65% source gate"})
    return {"schema_version": 1, "status": "SOURCE_READY" if not failures else "SOURCE_GATE_FAILED",
            "passed": not failures, "checks": checks, "failures": failures,
            "neural_operations": "not run", "audit_used_for_selection": False,
            "audit_policy": "Only training and validation shortcuts decide this gate. Any previously inspected held-out diagnostic must be retired from later independent neural audit."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    config = check_config(strict_json((HERE / "config.json").read_bytes()))
    frozen = strict_json((HERE / "frozen-baselines-v5.json").read_bytes())
    splits = {split: read_jsonl(args.data / (split + ".jsonl")) for split in config["limits"]}
    result = verdict(splits, frozen, config)
    write_json(args.report, result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
