# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote tensor controls for content-dependent same-verdict ranking."""
import os
from common import remote_only, dump
from verdict_contrast import scalar_contrast


def loss(chosen, rejected):
    remote_only()
    import torch
    if any(t.shape != (2,) or t.dtype != torch.float32 or t.device.type != "cpu" or not torch.isfinite(t).all() for t in (chosen, rejected)):
        raise ValueError("same-verdict contrast requires paired finite CPU FP32 logits")
    return torch.nn.functional.softplus(rejected.flip(0) - chosen).mean()


def smoke(root):
    remote_only()
    import torch
    observed = []
    for c, r in (([2., 2.], [-2., -2.]), ([2., 0.], [0., 2.]), ([2., 0.], [2., 0.]), ([0., 0.], [0., 0.])):
        chosen = torch.tensor(c, requires_grad=True)
        rejected = torch.tensor(r, requires_grad=True)
        value = loss(chosen, rejected)
        if abs(value.item() - scalar_contrast(c, r)) > 2e-5:
            raise ValueError("same-verdict tensor disagrees with primary BT/Softplus formula")
        value.backward()
        if not (chosen.grad < 0).all() or not (rejected.grad > 0).all() or not torch.isfinite(chosen.grad).all() or not torch.isfinite(rejected.grad).all():
            raise ValueError("same-verdict actual preference gradients reversed or nonfinite")
        observed.append({"chosen": c, "rejected": r, "loss": value.item(), "finite_correct_direction_gradients": True})
    negatives = []
    for name, c, r in (("shape", torch.zeros(1), torch.zeros(2)),
                       ("dtype", torch.zeros(2, dtype=torch.float64), torch.zeros(2)),
                       ("nonfinite", torch.tensor([float("nan"), 0.]), torch.zeros(2))):
        try:
            loss(c, r)
        except ValueError:
            negatives.append(name)
        else:
            raise ValueError("same-verdict negative control accepted")
    dump(root / "same-verdict-tensor-controls.json", {"passed": True, "controls": observed, "negative_controls": negatives,
        "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
        "scope": "Actual tensor/gradient controls, not independent quality or reference engine."})
