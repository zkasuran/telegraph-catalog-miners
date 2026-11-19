# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Replay fixed train-only source controls on development, never audit."""
import argparse
import json
from collections import Counter
from pathlib import Path

from baselines import replay
from data import FORMATS
from source import sha


def evaluate(root):
    manifest = json.loads((root / "freeze.json").read_bytes())
    baseline_manifest = json.loads((root / "baseline-manifest.json").read_bytes())
    baseline_raw = (root / "frozen-baselines.json").read_bytes()
    if (sha(baseline_raw) != baseline_manifest["frozen_baselines_sha256"]
            or baseline_manifest["training_sha256"] != manifest["splits"]["train"]["sha256"]
            or baseline_manifest["audit_scored"] is not False):
        raise ValueError("immutable control identity mismatch")
    fixed = json.loads(baseline_raw)
    reports = {}
    for split in ("train", "validation"):
        raw = (root / (split + ".json")).read_bytes()
        if sha(raw) != manifest["splits"][split]["sha256"]:
            raise ValueError("source split identity mismatch")
        rows = json.loads(raw)
        metrics = {}
        for format_name in FORMATS:
            formatted = [{**row, "chosen_title": row["formats"][format_name]["chosen"],
                          "rejected_title": row["formats"][format_name]["rejected"]} for row in rows]
            metrics[format_name] = replay(fixed, formatted, baseline_manifest["training_sha256"])
        reports[split] = metrics
    failures = [format_name + "/" + control + " validation shortcut above 0.65"
                for format_name, values in reports["validation"].items()
                for control, value in values.items() if value > .65]
    # Training replay is diagnostic. Coefficients already saw train, so that
    # accuracy is not an independent shortcut estimate or candidate selection.
    training = json.loads((root / "train.json").read_bytes())
    counts = Counter(row["chosen_title"] for row in training)
    rejected = Counter(row["rejected_title"] for row in training)
    structural = {"train_originals": len(training), "unique_rejected_fraction": len(rejected) / len(training),
                  "max_rejected_fraction": max(rejected.values()) / len(training),
                  "identical_fraction": sum(row["chosen_title"] == row["rejected_title"] for row in training) / len(training),
                  "repeated_chosen": sum(v > 1 for v in counts.values()), "normalized_overlap": manifest["normalized_overlap"]}
    if structural["unique_rejected_fraction"] < .8 or structural["max_rejected_fraction"] > .02 or structural["identical_fraction"] > .01:
        failures.append("source repetition/identity gate failed")
    report = {"schema_version": 1, "status": "DEVELOPMENT_SHORTCUT_FAIL" if failures else "DEVELOPMENT_SHORTCUT_PASS_REMOTE_GATE_PENDING",
              "training_allowed": not failures, "frozen_baseline_sha256": sha(baseline_raw),
              "training_sha256": baseline_manifest["training_sha256"], "metrics": reports, "structural": structural,
              "failures": failures, "audit_scored": False, "audit_used_for_selection": False,
              "controls_refit": False, "full_primary_dataset_gate_run": False,
              "limits": ["Integer source-control replay is not the full primary dataset gate or neural model quality.",
                         "No audit data bytes or metrics are opened by this command."]}
    path = root / "development-controls.json"
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError("immutable development replay changed")
    path.write_bytes(encoded)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-root", type=Path, required=True)
    result = evaluate(parser.parse_args().frozen_root)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["training_allowed"] else 1)
