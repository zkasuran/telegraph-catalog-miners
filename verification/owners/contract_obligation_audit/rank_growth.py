# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Expand our retained source-trained factors only on the public CPU runner."""
import math
import shutil

from contracts import file_sha, remote_only, strict_json, write_json


def growth_config(config):
    if config.get("peft_type") != "LORA" or config.get("r") != 8 or config.get("lora_alpha") != 32:
        raise ValueError("only the verified retained rank-eight checkpoint may grow")
    if config.get("bias", "none") != "none" or set(config.get("target_modules", [])) != {"Wqkv", "Wo", "Wi"}:
        raise ValueError("source-trained target layout changed")
    if any(config.get(key) for key in ("use_dora", "use_rslora", "rank_pattern", "alpha_pattern")):
        raise ValueError("the reference engine supports only one rank and scale")
    grown = {**config, "r": 16, "lora_alpha": 64}
    if grown["lora_alpha"]/grown["r"] != config["lora_alpha"]/config["r"]:
        raise ValueError("retained factor scaling changed")
    return grown


def growth_shapes(shape_a, shape_b):
    if (len(shape_a) != 2 or len(shape_b) != 2 or shape_a[0] != 8 or shape_b[1] != 8
            or any(type(value) is not int or not 1 <= value <= 8192 for value in (*shape_a, *shape_b))):
        raise ValueError("bounded retained factor shapes required")
    return (16, shape_a[1]), (shape_b[0], 16)


def require_learning(count):
    if type(count) is not int or not 1 <= count < 1 << 30:
        raise ValueError("expanded output factors must actually learn before audit")


def grow(parent, root, cfg, seed):
    remote_only()
    # Actual tensor work is inaccessible to the local source-only release gate.
    import torch
    from safetensors.torch import load_file, save_file
    config = strict_json((parent/"adapter_config.json").read_bytes())
    grown_config = growth_config(config)
    tensors = load_file(str(parent/"adapter_model.safetensors"), device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    old_names = set(tensors)
    pairs = 0
    for key in sorted(old_names):
        if not key.endswith(".lora_A.weight"):
            continue
        other = key.replace(".lora_A.weight", ".lora_B.weight")
        if other not in tensors:
            raise ValueError("retained source factor partner missing")
        a, b = tensors[key], tensors[other]
        shape_a, shape_b = growth_shapes(tuple(a.shape), tuple(b.shape))
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("nonfinite retained source factor")
        expanded_a = torch.empty(shape_a, dtype=a.dtype)
        bound = 1/math.sqrt(shape_a[1])
        expanded_a.uniform_(-bound, bound, generator=generator)
        expanded_a[:8] = a
        expanded_b = torch.zeros(shape_b, dtype=b.dtype)
        expanded_b[:, :8] = b
        if not torch.equal(expanded_a[:8], a) or not torch.equal(expanded_b[:, :8], b) or torch.count_nonzero(expanded_b[:, 8:]).item() != 0:
            raise ValueError("retained factors or zero new output factors changed")
        tensors[key], tensors[other] = expanded_a.contiguous(), expanded_b.contiguous()
        pairs += 1
    if pairs != cfg.num_hidden_layers*4 or set(tensors) != old_names:
        raise ValueError("every reference target layer must have both factors")
    target = root/"grown-source-parent"
    target.mkdir(exist_ok=True)
    save_file(tensors, str(target/"adapter_model.safetensors"), metadata={"format": "pt"})
    write_json(target/"adapter_config.json", grown_config)
    for path in parent.iterdir():
        if path.is_file() and path.name not in ("adapter_model.safetensors", "adapter_config.json"):
            shutil.copyfile(path, target/path.name)
    if file_sha(parent/"reward_head.pt") != file_sha(target/"reward_head.pt"):
        raise ValueError("retained reward head bytes changed during growth")
    write_json(root/"rank-growth.json", {"schema_version": 1, "parent_rank": 8, "rank": 16,
               "parent_alpha": 32, "alpha": 64, "scale": 4, "retained_factors_equal": True,
               "new_output_factors_zero": True, "actual_factor_pairs": pairs,
               "reward_head_sha256": file_sha(target/"reward_head.pt"),
               "parent_tensor_sha256": file_sha(parent/"adapter_model.safetensors"),
               "grown_tensor_sha256": file_sha(target/"adapter_model.safetensors"),
               "claim": "Retained factor values and scale preserved. No reference-engine bit identity or quality result is implied."})
    return target


def verify_trained(adapter, root, cfg):
    remote_only()
    import torch
    from safetensors.torch import load_file
    tensors = load_file(str(adapter/"adapter_model.safetensors"), device="cpu")
    factors = [value for name, value in tensors.items() if name.endswith(".lora_B.weight")]
    if len(factors) != cfg.num_hidden_layers*4 or any(value.ndim != 2 or value.shape[1] != 16 or not torch.isfinite(value).all() for value in factors):
        raise ValueError("trained expanded factor layout or finite values changed")
    learned = sum(torch.count_nonzero(value[:, 8:]).item() for value in factors)
    require_learning(learned)
    write_json(root/"rank-growth-trained.json", {"schema_version": 1, "rank": 16,
               "actual_new_output_nonzero": learned, "actual_output_factor_count": len(factors),
               "trained_tensor_sha256": file_sha(adapter/"adapter_model.safetensors"),
               "claim": "New output factors changed from zero during actual training. Not a quality or reference-engine result."})
