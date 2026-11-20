# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Fixed original-task bootstrap on existing training groups, never held-out data."""
import json

from data import HERE, digest

WARMUP_EPOCHS = 64
WARMUP_LR = .0001
WARMUP_STEPS = 256


def warmup_rows(manifest, train):
    from reciprocal import selected_rows
    # Selection rechecks the exact original manifest and complete training rows.
    # No source sentence, task instruction or title is rewritten.
    return selected_rows(manifest, train)


def warmup_plan(rows):
    from reciprocal import schedules
    plan = schedules(rows)["stochastic_batch"][:WARMUP_STEPS]
    if len(plan) != WARMUP_STEPS or any(len(batch) != 2 for batch in plan):
        raise ValueError("complete fixed original-task bootstrap required")
    return plan


def admit(metrics):
    if (metrics.get("n_eval") != 8 or metrics.get("independent_groups") != 4
            or metrics.get("correct") != 8 or metrics.get("pairwise_accuracy") != 1.
            or metrics.get("per_label") != {"A": 1., "B": 1.}):
        return False
    return True


def source_check():
    manifest = json.loads((HERE / "data/manifest.json").read_text())
    rows = warmup_rows(manifest, [r for g in manifest["splits"]["train"] for r in g["pairs"]])
    plan = warmup_plan(rows)
    appearances = [sum(i in batch for batch in plan) for i in range(len(rows))]
    if appearances != [64] * 8:
        raise ValueError("fixed bootstrap repetitions changed")
    return {"role_bootstrap_source": "PASS", "independent_training_groups": 4, "unique_training_pairs": 8,
            "training_presentations": sum(map(len, plan)), "optimizer_steps": WARMUP_STEPS,
            "plan_sha256": digest(json.dumps(plan, separators=(",", ":")).encode()),
            "original_complete_prompts": True, "validation_used": False, "audit_used": False}


if __name__ == "__main__":
    print(json.dumps(source_check()))
