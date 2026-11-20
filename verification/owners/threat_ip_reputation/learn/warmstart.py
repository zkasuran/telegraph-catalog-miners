# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Warm-start owned LoRA/head tensors remotely, preserving exact source bytes."""
import re

from contracts import FAMILY, FOUNDATION, remote_only

PARENT_SHA = "09d14be0a05e9ed2c2d728ba1083abeeb10e8d3b611c52a7fec6da30034ff0a6"


def package_tensor(parameter):
    match = re.fullmatch(r"base_model\.model\.model\.layers\.(\d+)\.(attn\.Wqkv|attn\.Wo|mlp\.Wi|mlp\.Wo)\.lora_([AB])(?:\.default)?\.weight", parameter)
    if match:
        return "lora." + match[1] + "." + match[2] + "." + match[3], True
    if parameter in ("base_model.model.classifier.modules_to_save.default.weight", "base_model.model.classifier.weight", "base_model.model.score.modules_to_save.default.weight", "base_model.model.score.weight"):
        return "head.weight", True
    if parameter in ("base_model.model.classifier.modules_to_save.default.bias", "base_model.model.classifier.bias", "base_model.model.score.modules_to_save.default.bias", "base_model.model.score.bias"):
        return "head.bias", False
    return None


def load_checkpoint(model, package):
    remote_only()
    import numpy as np
    import torch
    from float_runtime.lora_format import read_package
    from contracts import sha
    if sha(package.read_bytes()) != PARENT_SHA:
        raise ValueError("warm-start parent differs from actual preserved v2 checkpoint")
    metadata, tensors = read_package(package)
    if metadata["intent"] != FAMILY or metadata["class"] != "judgment" or metadata["foundation_hash"] != FOUNDATION["foundation_lora_fp32.onnx"] or metadata["tokenizer_hash"] != FOUNDATION["tokenizer.json"] or metadata["rank"] != "16" or metadata["alpha"] != "32" or metadata["max_tokens"] != "256":
        raise ValueError("warm-start parent layout/identity mismatch")
    used = set()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            mapped = package_tensor(name)
            if mapped is None:
                continue
            key, transpose = mapped
            if key in used:
                raise ValueError("warm-start tensor maps twice")
            used.add(key)
            value = np.ascontiguousarray(tensors[key].T if transpose else tensors[key])
            if tuple(parameter.shape) != value.shape:
                raise ValueError("warm-start tensor shape mismatch")
            parameter.copy_(torch.from_numpy(value).to(dtype=parameter.dtype, device=parameter.device))
    if used != set(tensors):
        raise ValueError("warm-start leaves an unmapped source tensor")
    return {"parent_sha256": PARENT_SHA, "tensors_loaded": len(used), "all_package_tensors_mapped_once": True,
            "base_weights_changed": False, "optimizer_state_resumed": False,
            "scope": "exact owned LoRA/head checkpoint, optimizer resets explicitly; no audit selection"}
