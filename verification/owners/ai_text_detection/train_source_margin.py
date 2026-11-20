# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Guarded tensor source-margin consistency; no runtime transform or relabelling."""
import os
from common import remote_only, dump
from source_margin import scalar_consistency


def consistency(chosen, rejected, source_logit, human, policy):
    remote_only()
    import torch
    if (type(human) is not int or human not in (0, 1) or
            any(t.shape != chosen.shape or t.ndim != 1 or t.numel() != 1 or
                t.device.type != "cpu" or t.dtype != torch.float32 or not torch.isfinite(t).all()
                for t in (chosen, rejected, source_logit))):
        raise ValueError("source-margin tensor shape/device/finite/role contract failed")
    signed = (2 * human - 1) * source_logit.detach()
    target = (2 * torch.nn.functional.softplus(signed, beta=1, threshold=20)).clamp(policy["target_min"], policy["target_max"])
    loss = torch.nn.functional.smooth_l1_loss(chosen - rejected, target, beta=policy["beta"], reduction="mean")
    if not torch.isfinite(loss).all() or target.requires_grad:
        raise ValueError("source-margin target was not finite and detached")
    return loss


def smoke(policy, root):
    remote_only()
    import torch
    observed = []
    for c, r, z, human in ((0., 0., 2., 1), (0., 0., -2., 0), (1., -1., -2., 1), (1000., -1000., 1000., 1), (0., 0., -1000., 1)):
        chosen = torch.tensor([c], requires_grad=True)
        rejected = torch.tensor([r], requires_grad=True)
        teacher = torch.tensor([z], requires_grad=True)
        loss = consistency(chosen, rejected, teacher, human, policy)
        if abs(loss.item() - scalar_consistency(c, r, z, human, policy)) > 2e-4:
            raise ValueError("actual source-margin loss disagreed with primary scalar formula")
        loss.backward()
        if teacher.grad is not None or not torch.isfinite(chosen.grad).all() or not torch.isfinite(rejected.grad).all():
            raise ValueError("source-margin teacher-detach/finite-gradient control failed")
        if c == r == 0 and not chosen.grad.item() < 0 < rejected.grad.item():
            raise ValueError("source-margin preference direction reversed")
        observed.append({"student_margin": c-r, "teacher_logit": z, "human": human, "loss": loss.item(),
                         "teacher_gradient_absent": teacher.grad is None})
    refused = []
    for name in ("shape", "nonfinite", "dtype", "origin"):
        c, r, z, h = torch.zeros(1), torch.zeros(1), torch.zeros(1), 1
        if name == "shape":
            z = torch.zeros(2)
        elif name == "nonfinite":
            z[0] = float("nan")
        elif name == "dtype":
            z = z.double()
        else:
            h = True
        try:
            consistency(c, r, z, h, policy)
        except ValueError:
            refused.append(name)
        else:
            raise ValueError("source-margin negative control accepted: " + name)
    dump(root / "source-margin-tensor-controls.json", {"passed": True, "controls": observed, "negative_controls": refused,
        "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
        "scope": "Actual tensor/gradient controls, not model or independent runtime quality."})
