# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Frozen-encoder reward-head experiment. Source checks never import neural code."""
import json
from common import sha, identity
from coupled_contract import source_groups
from verdict_contrast import check_pair
from checkpoint_contract import check_checkpoint


def headfit_policy(recipe):
    expected = {"scope": "training_only", "encoder_frozen": True, "origin_readout_frozen": True,
                "trainable_tensors": 2, "steps": 256, "validation_interval": 8, "learning_rate": 0.001,
                "clip_max_norm": 1.0, "cache_max_bytes": 16777216, "all_original_objective_terms_retained": True,
                "audit_used_for_selection": False, "runtime_transform": False,
                "checkpoint_file": "recovery/curriculum-improved-checkpoint.safetensors",
                "checkpoint_sha256": "4a133056f5be55091fa52348a4b0b70a4f14ef405b40ae3473138affcfa8ef3c",
                "readout_file": "recovery/curriculum-improved-readout.safetensors",
                "readout_sha256": "8c637998c076ade16a0fa98f866786e1e6b28d3ea554f180c36bf0d6efbf87a5",
                "producer_run_id": 44860695142, "producer_commit": "d7c7030531ed8106c7397da8508b01509fc961e2", "selected_epoch": 3}
    spec = recipe.get("head_fit")
    if (spec != expected or any(type(spec[k]) is not int for k in ("trainable_tensors", "steps", "validation_interval", "cache_max_bytes", "producer_run_id", "selected_epoch")) or
            any(spec[k] is not True for k in ("encoder_frozen", "origin_readout_frozen", "all_original_objective_terms_retained")) or
            spec["audit_used_for_selection"] is not False or spec["runtime_transform"] is not False):
        raise ValueError("fixed train-only frozen-encoder head fit required")
    return dict(expected)


def effective_recipe(recipe):
    if not recipe.get("head_fit"):
        return dict(recipe)
    policy = headfit_policy(recipe)
    return {**recipe, "checkpoint_file": policy["checkpoint_file"], "checkpoint_sha256": policy["checkpoint_sha256"],
            "origin_checkpoint_file": policy["readout_file"], "origin_checkpoint_sha256": policy["readout_sha256"]}


def check_start(raw, recipe):
    policy = headfit_policy(recipe)
    checked = check_checkpoint(raw, policy["checkpoint_sha256"], recipe["parent_source_manifest_sha256"])
    if checked["descriptors"] != 178:
        raise ValueError("preserved head-fit checkpoint inventory changed")
    return {**checked, "producer_run_id": policy["producer_run_id"], "selected_epoch": 3, "payload_interpreted": False}


def plan(rows, recipe):
    policy = headfit_policy(recipe)
    if len(rows) != 1008 or any(r["split"] != "train" or r["group_id"] != identity(r["source_id"]) or r["margin"] != 0 for r in rows):
        raise ValueError("unchanged complete zero-margin training population required")
    groups = source_groups(rows)
    for group in groups:
        check_pair(group)
    indexes = {r["text_id"]: i for i, r in enumerate(rows)}
    if len(indexes) != 1008 or len(groups) != 504:
        raise ValueError("unique complete paired-source population required")
    binding = {"policy": policy, "training_rows_sha256": sha(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()),
               "training_rows": 1008, "source_view_groups": 504, "group_indices": [[indexes[r["text_id"]] for r in g] for g in groups],
               "new_data": 0, "audit_used": False}
    return binding
