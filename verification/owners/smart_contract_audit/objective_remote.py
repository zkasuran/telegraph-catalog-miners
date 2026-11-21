# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Actual differentiable joint objective and controls, authorized CPU runner only."""
import json
from pathlib import Path

from policy import remote_only
from objective import scalar_loss


def loss(chosen, rejected, margin):
    remote_only()  # Before Torch import or tensor operation.
    import torch
    import torch.nn.functional as F
    if chosen.ndim != 1 or chosen.shape != rejected.shape or chosen.shape != margin.shape or not 0 < chosen.numel() <= 16:
        raise ValueError("aligned bounded factual preference scores required")
    if not torch.isfinite(chosen).all() or not torch.isfinite(rejected).all() or not torch.isfinite(margin).all():
        raise ValueError("finite training preference scores required")
    ranking = -F.logsigmoid(chosen.float() - rejected.float() - margin.float()).mean()
    calibration = (F.binary_cross_entropy_with_logits(chosen.float(), torch.ones_like(chosen.float())) +
                   F.binary_cross_entropy_with_logits(rejected.float(), torch.zeros_like(rejected.float()))) / 2
    return (ranking + calibration) / 2


def check(output_root):
    remote_only()
    import torch
    chosen = torch.tensor([0.0, 2.0, -2.0], requires_grad=True)
    rejected = torch.tensor([0.0, -2.0, 2.0], requires_grad=True)
    margin = torch.zeros(3)
    actual = loss(chosen, rejected, margin)
    expected = sum(scalar_loss(a, b) for a, b in zip(chosen.detach().tolist(), rejected.detach().tolist())) / 3
    if abs(actual.item() - expected) > 1e-6:
        raise ValueError("actual Torch objective differs from independent stable scalar formula")
    actual.backward()
    if not torch.isfinite(chosen.grad).all() or not torch.isfinite(rejected.grad).all() or not (chosen.grad < 0).all() or not (rejected.grad > 0).all():
        raise ValueError("actual objective gradient finite/direction control failed")
    negative = []
    for name, left, right in (("shape mismatch", torch.zeros(2), torch.zeros(1)), ("nonfinite", torch.tensor([float('nan')]), torch.zeros(1))):
        try:
            loss(left, right, torch.zeros_like(left))
        except ValueError:
            negative.append(name)
        else:
            raise ValueError("objective accepted " + name)
    report = {"actual_torch_version": torch.__version__, "native_loss": actual.item(), "independent_scalar_loss": expected,
              "gradient_direction_and_finiteness_passed": True, "chosen_gradients": chosen.grad.tolist(), "rejected_gradients": rejected.grad.tolist(),
              "negative_controls": negative, "model_quality_claimed": False, "audit_or_validation_gradients_used": False}
    (Path(output_root) / "objective-runtime-controls.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
