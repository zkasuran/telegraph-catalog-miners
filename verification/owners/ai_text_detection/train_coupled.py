# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only counterpart objective, no label-specific or serving transform."""
import os
from common import remote_only, dump
from coupled_contract import scalar_coupled


def coupled_loss(losses):
    remote_only()
    import torch
    if len(losses) != 2 or any(x.numel() != 1 or x.device.type != "cpu" or x.dtype != torch.float32 or
                             not torch.isfinite(x).all() or x.item() < 0 for x in losses):
        raise ValueError("two finite nonnegative CPU FP32 counterpart losses required")
    a, b = losses
    return (a + b) / 2 + 0.25 * torch.abs(a - b)


def smoke(root):
    remote_only()
    import torch
    observed = []
    for a, b, gradients in ((1.0, 3.0, (0.25, 0.75)), (3.0, 1.0, (0.75, 0.25)), (2.0, 2.0, (0.5, 0.5))):
        inputs = [torch.tensor(x, dtype=torch.float32, requires_grad=True) for x in (a, b)]
        loss = coupled_loss(inputs)
        loss.backward()
        if abs(loss.item() - scalar_coupled(a, b)) > 1e-6 or any(abs(x.grad.item() - g) > 1e-6 for x, g in zip(inputs, gradients)):
            raise ValueError("actual coupled value/gradient control failed")
        observed.append({"input_losses": [a, b], "loss": loss.item(), "gradients": [x.grad.item() for x in inputs]})
    for values in ([torch.tensor(float("nan")), torch.tensor(1.0)], [torch.tensor(-1.0), torch.tensor(1.0)]):
        try:
            coupled_loss(values)
        except ValueError:
            continue
        raise ValueError("coupled invalid-loss refusal did not fail")
    dump(root / "coupled-tensor-controls.json", {"passed": True, "actual_controls": observed,
        "negative_controls": 2, "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
        "scope": "Actual tensor value/gradient controls, not model quality or node runtime."})
