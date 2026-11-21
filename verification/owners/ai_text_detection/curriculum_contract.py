# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded replay of existing canonical training formats, never new source rows."""
import json
import random

from common import sha, identity
from coupled_contract import source_groups
from verdict_contrast import check_pair


def curriculum_policy(recipe):
    expected = {"scope": "existing_training_only", "unique_training_rows": 1008,
                "unique_bare_rows": 16, "bare_source_views": 8, "rounds_per_epoch": 4,
                "views_per_update": 4, "additional_updates_per_epoch": 8, "seed": 2042,
                "learning_rate": "existing_hp_lr", "clip_max_norm": 1.0,
                "original_phases_and_scheduler_retained": True, "old_losses_and_weights_retained": True,
                "validation_or_audit_replay": False, "runtime_transform": False}
    spec = recipe.get("format_curriculum")
    if (spec != expected or any(type(spec[k]) is not int for k in ("unique_training_rows", "unique_bare_rows", "bare_source_views",
            "rounds_per_epoch", "views_per_update", "additional_updates_per_epoch", "seed")) or
            spec["validation_or_audit_replay"] is not False or spec["runtime_transform"] is not False or
            spec["original_phases_and_scheduler_retained"] is not True or spec["old_losses_and_weights_retained"] is not True):
        raise ValueError("fixed bounded train-only curriculum required")
    if recipe["grad_accum"] != 8 or recipe["bare_loss_weight"] != 16 or recipe["lr"] != 1e-5:
        raise ValueError("curriculum must retain accumulation, format weights and learning rate")
    return dict(expected)


def plan(rows, recipe):
    policy = curriculum_policy(recipe)
    if len(rows) != 1008 or any(r["split"] != "train" or r["group_id"] != identity(r["source_id"]) for r in rows):
        raise ValueError("curriculum requires unchanged complete training population")
    groups = source_groups(rows)
    for group in groups:
        check_pair(group)
    bare = [group for group in groups if group[0]["style"] == "bare_label"]
    if len(bare) != 8 or len({g[0]["group_id"] for g in bare}) != 8:
        raise ValueError("exact eight existing independent bare training views required")
    flat = [r for group in bare for r in group]
    if len({r["text_id"] for r in flat}) != 16 or sum(r["label"] == "Human" for r in flat) != 8 or sum(r["label"] == "AI-generated" for r in flat) != 8:
        raise ValueError("curriculum must retain unique balanced canonical comparisons")
    binding = {"policy": policy, "training_rows_sha256": sha(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()),
               "bare_rows_sha256": sha(json.dumps(bare, sort_keys=True, ensure_ascii=False).encode()),
               "view_ids": [g[0]["view_id"] for g in bare], "source_groups": [g[0]["group_id"] for g in bare],
               "unique_training_rows": 1008, "unique_bare_rows": 16, "presentations_per_epoch": 64,
               "additional_optimizer_updates_per_epoch": 8, "new_source_or_preference_rows": 0,
               "preference_authorship": "Constructed source-backed training comparisons, not human-reviewed votes or independent audit."}
    return bare, binding


def rounds(groups, epoch, policy):
    if type(epoch) is not int or not 0 <= epoch < 4 or len(groups) != 8:
        raise ValueError("bounded four-epoch complete-view curriculum required")
    result = []
    for index in range(policy["rounds_per_epoch"]):
        ordered = list(groups)
        random.Random(policy["seed"] + epoch * 100 + index).shuffle(ordered)
        result.append(ordered)
    return result
