# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only continuation in a larger supported LoRA subspace, no score mapping."""
import os

from common import remote_only, dump


def expand(saved, initialized, policy):
    remote_only()
    import torch
    if set(saved) != set(initialized):
        raise ValueError("capacity learned-state key mismatch")
    rank, new_rank = policy["old_rank"], policy["new_rank"]
    if (rank, new_rank, policy["old_alpha"], policy["new_alpha"]) != (8, 16, 32, 64):
        raise ValueError("capacity scale or rank policy mismatch")
    result = {}
    for name, old in saved.items():
        current = initialized[name]
        if any(t.device.type != "cpu" or t.dtype != torch.float32 or not torch.isfinite(t).all() for t in (old, current)):
            raise ValueError("capacity expansion requires finite CPU FP32 state")
        if ".lora_A." in name:
            if old.ndim != 2 or current.shape != (new_rank, old.shape[1]) or old.shape[0] != rank:
                raise ValueError("capacity A shape mismatch")
            value = current.detach().clone()
            value[:rank] = old
            if not torch.count_nonzero(value[rank:]).item():
                raise ValueError("additional capacity A rows are all zero")
        elif ".lora_B." in name:
            if old.ndim != 2 or current.shape != (old.shape[0], new_rank) or old.shape[1] != rank:
                raise ValueError("capacity B shape mismatch")
            if torch.count_nonzero(current[:, rank:]).item():
                raise ValueError("additional capacity B columns were not initialized to zero")
            value = torch.zeros_like(current)
            value[:, :rank] = old
        else:
            if old.shape != current.shape:
                raise ValueError("capacity expansion changed reward-head shape")
            value = old.detach().clone()
        result[name] = value
    maximum_delta_error = 0.0
    paired = 0
    for name, old_a in saved.items():
        if ".lora_A." not in name:
            continue
        b_name = name.replace(".lora_A.", ".lora_B.")
        if b_name not in saved:
            raise ValueError("capacity expansion lost B counterpart")
        before = (saved[b_name] @ old_a) * (policy["old_alpha"] / rank)
        after = (result[b_name] @ result[name]) * (policy["new_alpha"] / new_rank)
        if not torch.equal(result[name][:rank], old_a) or not torch.equal(result[b_name][:, :rank], saved[b_name]):
            raise ValueError("capacity expansion changed original tensor slices")
        error = torch.max(torch.abs(before - after)).item()
        maximum_delta_error = max(maximum_delta_error, error)
        if not torch.allclose(before, after, atol=1e-6, rtol=0):
            raise ValueError("capacity expansion changed the initial FP32 delta")
        paired += 1
    if not paired:
        raise ValueError("capacity expansion has no LoRA matrices")
    return result, {"retained_rank_slices_bits_identical": True, "reward_head_bits_identical": True,
                    "new_b_columns_zero": True, "scale_before": 4, "scale_after": 4,
                    "paired_modules": paired, "maximum_actual_delta_error": maximum_delta_error,
                    "delta_absolute_tolerance": 1e-6, "whole_state_bytes_identical": False}


def smoke(policy, root):
    remote_only()
    import torch
    a, b = "fixture.lora_A.weight", "fixture.lora_B.weight"
    old = {a: torch.arange(24, dtype=torch.float32).reshape(8, 3) / 20,
           b: torch.arange(16, dtype=torch.float32).reshape(2, 8) / 30,
           "fixture.classifier.weight": torch.ones(1, 3), "fixture.classifier.bias": torch.zeros(1)}
    current = {a: torch.arange(48, dtype=torch.float32).reshape(16, 3) / 50,
               b: torch.zeros(2, 16), "fixture.classifier.weight": torch.zeros(1, 3), "fixture.classifier.bias": torch.ones(1)}
    expanded, report = expand(old, current, policy)
    learnable_b = expanded[b].clone().requires_grad_()
    loss = ((learnable_b @ expanded[a]) ** 2).sum()
    loss.backward()
    if not torch.count_nonzero(learnable_b.grad[:, 8:]).item() or not torch.isfinite(learnable_b.grad).all():
        raise ValueError("additional B capacity did not receive finite fixture gradients")
    refused = []
    for control in ("keys", "shape", "nonfinite", "dtype", "nonzero_new_b", "scale"):
        bad_old = {k: t.clone() for k, t in old.items()}
        bad_current = {k: t.clone() for k, t in current.items()}
        bad_policy = dict(policy)
        if control == "keys":
            del bad_old[b]
        elif control == "shape":
            bad_current[a] = torch.ones(16, 4)
        elif control == "nonfinite":
            bad_old[a][0, 0] = float("nan")
        elif control == "dtype":
            bad_current[a] = bad_current[a].double()
        elif control == "nonzero_new_b":
            bad_current[b][0, 8] = 1
        else:
            bad_policy["new_alpha"] = 32
        try:
            expand(bad_old, bad_current, bad_policy)
        except ValueError:
            refused.append(control)
        else:
            raise ValueError("capacity negative control was accepted: " + control)
    dump(root / "capacity-tensor-controls.json", {**report, "passed": True, "fixture_only": True,
        "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
        "additional_b_fixture_gradient_nonzero": True, "negative_controls": refused})
