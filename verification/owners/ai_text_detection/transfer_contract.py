# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Train-only format-balanced replay from the unchanged accepted population."""
import json
import random

from common import sha, identity, FORMATS
from coupled_contract import source_groups
from verdict_contrast import check_pair


def transfer_policy(recipe):
    expected = {"scope": "existing_training_only", "formats": ["json", "prose", "bare_label"],
                "views_per_format": 8, "rounds_per_epoch": 4, "views_per_update": 4, "extra_updates_per_epoch": 24,
                "replay_pair_weight": 1.0, "seed": 3042, "clip_max_norm": 1.0, "learning_rate": "existing_hp_lr",
                "replaces_bare_only_replay": True, "original_source_losses_weights_scheduler_retained": True,
                "validation_or_audit_replayed": False, "runtime_transform": False}
    spec = recipe.get("format_transfer_curriculum")
    if (spec != expected or any(type(spec[k]) is not int for k in ("views_per_format", "rounds_per_epoch", "views_per_update", "extra_updates_per_epoch", "seed")) or
            spec["replaces_bare_only_replay"] is not True or spec["original_source_losses_weights_scheduler_retained"] is not True or
            spec["validation_or_audit_replayed"] is not False or spec["runtime_transform"] is not False or recipe["grad_accum"] != 8):
        raise ValueError("fixed bounded format-balanced replay required")
    return dict(expected)


def plan(rows, recipe):
    policy = transfer_policy(recipe)
    if len(rows) != 1008 or any(r["split"] != "train" or r["group_id"] != identity(r["source_id"]) for r in rows):
        raise ValueError("unchanged complete legitimate training rows required")
    population = source_groups(rows)
    for group in population:
        check_pair(group)
    # Retain every old bare view, then choose other format groups by fixed hashes.
    selected = {"bare_label": [g for g in population if g[0]["style"] == "bare_label"]}
    used = {g[0]["group_id"] for g in selected["bare_label"]}
    if len(selected["bare_label"]) != 8 or len(used) != 8:
        raise ValueError("all eight existing bare source views required")
    for style in ("json", "prose"):
        selected[style] = []
        candidates = sorted((g for g in population if g[0]["style"] == style), key=lambda g: (g[0]["group_id"], g[0]["view_id"]))
        for group in candidates:
            if group[0]["group_id"] in used:
                continue
            selected[style].append(group)
            used.add(group[0]["group_id"])
            if len(selected[style]) == 8:
                break
        if len(selected[style]) != 8:
            raise ValueError("eight independent source groups for every replay format required")
    ordered = {style: selected[style] for style in FORMATS}
    binding = {"policy": policy, "unique_training_rows": 1008, "unique_replayed_rows": 48,
               "independent_replayed_groups": len(used), "training_presentations_per_epoch": 192,
               "additional_optimizer_updates_per_epoch": 24, "new_source_or_preference_rows": 0,
               "training_rows_sha256": sha(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()),
               "selected_rows_sha256": sha(json.dumps(ordered, sort_keys=True, ensure_ascii=False).encode()),
               "view_ids": {style: [g[0]["view_id"] for g in ordered[style]] for style in FORMATS}}
    return ordered, binding


def schedule(groups, epoch, policy):
    if type(epoch) is not int or not 0 <= epoch < 4 or set(groups) != set(FORMATS) or any(len(g) != 8 for g in groups.values()):
        raise ValueError("bounded complete three-format epoch required")
    result = []
    for round_index in range(4):
        ordered = {}
        for index, style in enumerate(FORMATS):
            ordered[style] = list(groups[style])
            random.Random(policy["seed"] + epoch * 100 + round_index * 10 + index).shuffle(ordered[style])
        cursors = {style: 0 for style in FORMATS}
        for update in range(6):
            dominant = FORMATS[update % 3]
            batch = []
            for style in FORMATS:
                count = 2 if style == dominant else 1
                start = cursors[style]
                batch.extend(ordered[style][start:start+count])
                cursors[style] += count
            result.append({"round": round_index, "groups": batch})
        if set(cursors.values()) != {8}:
            raise ValueError("balanced round lost a source view")
    return result
