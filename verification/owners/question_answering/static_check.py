# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Standalone source-only release contracts, no fitting or neural imports."""
import ast
import json
from pathlib import Path

from data import RECIPE, digest
from licensing import licence_metadata


def main():
    here = Path(__file__).resolve().parent
    for path in here.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    contract = json.loads((here / "contract.json").read_text())
    if contract["recipe"] != RECIPE or contract["generation"] != 2:
        raise ValueError("frozen recipe changed")
    from debias_policy import GROUP, WEIGHT
    objective = contract["training_objective"]
    if objective["optimizer_group"] != GROUP or WEIGHT != .25 or objective["fit_split"] != "train" or objective["serving_remap"] is not False:
        raise ValueError("train-only objective contract changed")
    if contract["quality"] != {"minimum_accuracy": .8, "minimum_subset_accuracy": .7, "minimum_untrained_gain": .05,
                               "minimum_shortcut_gain": .10, "maximum_shortcut_accuracy": .65}:
        raise ValueError("quality gates changed")
    if {name: pin["rows"] for name, pin in contract["splits"].items()} != {"train": 768, "validation": 128, "audit": 64}:
        raise ValueError("independent source counts changed")
    if digest((here / "baselines/frozen-controls.json").read_bytes()) != contract["frozen_controls_sha256"]:
        raise ValueError("immutable training baseline changed")
    if digest((here / "exposed-exclusions.json").read_bytes()) != contract["exposed_exclusion_sha256"]:
        raise ValueError("exposure exclusion changed")
    if digest((here / "development-freeze.json").read_bytes()) != contract["development_freeze_sha256"]:
        raise ValueError("development freeze changed")
    licence_metadata()
    print("Standalone QA recipe, source freezes, exposure exclusion and separate licence gates: PASS")


if __name__ == "__main__":
    main()
