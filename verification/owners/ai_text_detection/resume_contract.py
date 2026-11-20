# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Exact selected rank16 continuation, source-only until the remote guard."""
import json
import struct

from checkpoint_contract import check_checkpoint

CHECKPOINT = "05d0d605cbe0e4f05af43f0e15562fbba6d3d5b41d45846b38b495f51b230b6f"
READOUT = "47ae66f72ff3d6b984d9550b8068a84611f84dabe7797fd1d6e5afcb9a51b1ab"
COMMIT = "3e140852eae8ded934d3bb29dc18e13f5693a6a6"


def resume_policy(recipe):
    expected = {"rank": 16, "alpha": 64, "producer_run_id": 44762535076,
                "producer_commit": COMMIT, "selected_epoch": 0,
                "checkpoint_sha256": CHECKPOINT, "readout_sha256": READOUT,
                "direct_load": True, "reexpand": False, "audit_used": False}
    spec = recipe.get("rank16_resume")
    if (spec != expected or spec["direct_load"] is not True or spec["reexpand"] is not False or spec["audit_used"] is not False or
            any(type(spec[k]) is not int for k in ("rank", "alpha", "producer_run_id", "selected_epoch")) or
            recipe.get("checkpoint_sha256") != CHECKPOINT or recipe.get("origin_checkpoint_sha256") != READOUT or
            recipe.get("lora_rank") != 16 or recipe.get("lora_alpha") != 64 or recipe.get("lr") != 1e-5 or
            type(recipe["lr"]) not in (int, float)):
        raise ValueError("fixed exact selected rank16 lower-rate resume required")
    return dict(expected)


def source_resume(raw, recipe):
    policy = resume_policy(recipe)
    checked = check_checkpoint(raw, policy["checkpoint_sha256"], recipe["parent_source_manifest_sha256"])
    n = struct.unpack("<Q", raw[:8])[0]
    descriptors = json.loads(raw[8:8 + n])
    descriptors.pop("__metadata__", None)
    matrices = {k: v for k, v in descriptors.items() if ".lora_A." in k or ".lora_B." in k}
    if len(matrices) != 176 or len(descriptors) != 178:
        raise ValueError("selected rank16 descriptor inventory changed")
    for name, row in matrices.items():
        shape = row.get("shape", [])
        axis = 0 if ".lora_A." in name else 1
        if len(shape) != 2 or shape[axis] != 16 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError("selected rank16 matrix shape changed")
        counterpart = name.replace(".lora_A.", ".lora_B.") if axis == 0 else name.replace(".lora_B.", ".lora_A.")
        if counterpart not in matrices:
            raise ValueError("selected rank16 counterpart missing")
    return {**checked, "rank": 16, "alpha": 64, "producer_run_id": policy["producer_run_id"],
            "selected_epoch": 0, "payload_interpreted": False, "optimizer_state_available": False}
