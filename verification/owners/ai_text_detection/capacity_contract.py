# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only rank-expansion policy and opaque checkpoint descriptor controls."""
import json
import struct

from checkpoint_contract import check_checkpoint


def capacity_policy(recipe):
    expected = {"old_rank": 8, "new_rank": 16, "old_alpha": 32, "new_alpha": 64,
                "new_a": "seeded_primary_initialization", "new_b": "zero",
                "old_slices_retained": True, "head_retained": True,
                "runtime_transform": False, "audit_used": False}
    spec = recipe.get("capacity_expansion")
    if (spec != expected or any(type(spec[k]) is not int for k in ("old_rank", "new_rank", "old_alpha", "new_alpha")) or
            spec["old_slices_retained"] is not True or spec["head_retained"] is not True or
            spec["runtime_transform"] is not False or spec["audit_used"] is not False or
            recipe.get("lora_rank") != 16 or recipe.get("lora_alpha") != 64 or
            type(recipe["lora_rank"]) is not int or type(recipe["lora_alpha"]) is not int):
        raise ValueError("fixed bounded rank8-to-rank16 continuation required")
    if spec["old_alpha"] / spec["old_rank"] != spec["new_alpha"] / spec["new_rank"]:
        raise ValueError("rank expansion changed the actual LoRA scale")
    return dict(expected)


def source_capacity(raw, recipe):
    policy = capacity_policy(recipe)
    checked = check_checkpoint(raw, recipe["checkpoint_sha256"], recipe["parent_source_manifest_sha256"])
    size = struct.unpack("<Q", raw[:8])[0]
    descriptors = json.loads(raw[8:8 + size])
    descriptors.pop("__metadata__", None)
    matrices = {k: v for k, v in descriptors.items() if ".lora_A." in k or ".lora_B." in k}
    heads = {k: v for k, v in descriptors.items() if k not in matrices}
    if len(matrices) != 176 or len(heads) != 2:
        raise ValueError("capacity continuation changed the original matrix/head inventory")
    for name, desc in matrices.items():
        shape = desc.get("shape", [])
        axis = 0 if ".lora_A." in name else 1
        if len(shape) != 2 or shape[axis] != policy["old_rank"] or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError("capacity continuation requires original rank8 descriptors")
        opposite = name.replace(".lora_A.", ".lora_B.") if axis == 0 else name.replace(".lora_B.", ".lora_A.")
        if opposite not in matrices:
            raise ValueError("capacity continuation lost a paired matrix")
    return {**checked, "old_rank": 8, "new_rank": 16, "scale": 4,
            "original_lora_descriptors": len(matrices), "retained_head_descriptors": len(heads),
            "new_model_loaded_locally": False, "expanded_state_built": False}
