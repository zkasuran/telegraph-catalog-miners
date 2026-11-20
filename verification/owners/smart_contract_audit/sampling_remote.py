# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Actual Torch sampler controls on the authorized public CPU runner only."""
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path

from policy import remote_only
from sampling import weights


def check(rows, frozen, output_root, seed):
    remote_only()  # Before Torch import or runtime operations.
    import torch
    from torch.utils.data import WeightedRandomSampler
    if torch.__version__ != "2.15.0+cpu":
        raise ValueError("pinned CPU Torch version required")
    actual = weights(rows, frozen)
    draws = 12000
    first = list(WeightedRandomSampler(actual, draws, replacement=True, generator=torch.Generator().manual_seed(seed)))
    second = list(WeightedRandomSampler(actual, draws, replacement=True, generator=torch.Generator().manual_seed(seed)))
    if first != second or any(type(i) is not int or not 0 <= i < len(rows) for i in first):
        raise ValueError("actual sampler repeat or index identity failed")
    counts = Counter(rows[i]["template_group"] for i in first)
    expected = draws / len(frozen["coefficients"])
    if set(counts) != set(frozen["coefficients"]) or any(not 0.7 * expected <= count <= 1.3 * expected for count in counts.values()):
        raise ValueError("actual weighted sampler failed bounded group-distribution control")
    negative = []
    for name, bad_weights, number in (("zero samples", [1, 1], 0), ("boolean samples", [1, 1], True), ("negative weight", [-1, 1], 2)):
        try:
            list(WeightedRandomSampler(bad_weights, number, replacement=True))
        except (ValueError, RuntimeError) as error:
            negative.append({"control": name, "observed_exception": type(error).__name__})
        else:
            raise ValueError("actual sampler accepted " + name)
    report = {"torch_version": torch.__version__, "actual_draws": draws, "actual_group_counts": dict(counts),
              "repeat_indices_identical": True, "negative_controls": negative,
              "class_source_sha256": hashlib.sha256(inspect.getsource(WeightedRandomSampler).encode()).hexdigest(),
              "training_views": len(rows), "training_groups": len(counts), "seed": seed,
              "audit_used_for_fitting": False, "model_quality_claimed": False}
    (Path(output_root) / "sampling-runtime-controls.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
