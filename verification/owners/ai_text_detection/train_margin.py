# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only source-paired verdict-margin regularization, not score rescaling."""
import os
from common import remote_only, dump
from margin_contract import scalar_margin


def alignment(chosen, rejected):
    remote_only()
    import torch
    if (chosen.shape != rejected.shape or chosen.ndim != 1 or chosen.numel() != 2 or
            any(v.device.type != "cpu" or v.dtype != torch.float32 or not torch.isfinite(v).all() for v in (chosen, rejected))):
        raise ValueError("paired finite CPU FP32 verdict scores required")
    margins = chosen - rejected
    if not torch.isfinite(margins).all():
        raise ValueError("nonfinite computed source margins")
    result = torch.nn.functional.smooth_l1_loss(margins[0:1], margins[1:2], beta=1.0, reduction="mean")
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite margin alignment loss")
    return result


def smoke(root):
    remote_only()
    import torch
    controls = []
    for a, b, expected_grad in ((4.0, 1.0, (1.0, -1.0)), (1.0, 4.0, (-1.0, 1.0)), (2.0, 2.0, (0.0, 0.0)), (0.5, 0.0, (0.5, -0.5))):
        chosen = torch.tensor([a, b], dtype=torch.float32, requires_grad=True)
        rejected = torch.zeros_like(chosen)
        loss = alignment(chosen, rejected)
        loss.backward()
        if abs(loss.item() - scalar_margin(a, b)) > 1e-6 or any(abs(x - y) > 1e-6 for x, y in zip(chosen.grad.tolist(), expected_grad)):
            raise ValueError("actual margin value/gradient control failed")
        controls.append({"margins": [a, b], "loss": loss.item(), "gradients": chosen.grad.tolist()})
    refused = []
    for name, a, b in (("shape", torch.zeros(1), torch.zeros(2)),
                       ("nonfinite", torch.tensor([float("nan"), 1.0]), torch.zeros(2)),
                       ("dtype", torch.zeros(2, dtype=torch.float64), torch.zeros(2))):
        try:
            alignment(a, b)
        except ValueError:
            refused.append(name)
        else:
            raise ValueError("margin negative control accepted " + name)
    dump(root / "margin-tensor-controls.json", {"passed": True, "actual_controls": controls,
        "negative_controls": refused,
        "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
        "scope": "Actual training-only tensor controls, not model or reference-engine quality."})
