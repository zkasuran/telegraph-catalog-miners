# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Opt-in paired-format neural objective. Every tensor operation is remote-only."""
from contextlib import contextmanager
import os

from contracts import FAMILY, remote_only, write_json
from paired_objective import MAX_GROUP_BATCH, grouped_collator, scalar_loss


def tensor_loss(chosen, rejected, margin, penalty):
    remote_only()
    scalar_loss(0, 0, penalty)  # Strict finite coefficient contract before tensor access.
    import torch
    import torch.nn.functional as functional
    if (chosen.ndim != 1 or chosen.shape != rejected.shape or chosen.shape != margin.shape
            or not 2 <= chosen.numel() <= 2 * MAX_GROUP_BATCH or chosen.numel() % 2
            or any(value.device.type != "cpu" for value in (chosen, rejected, margin))
            or not 0 <= penalty <= 1):
        raise ValueError("paired CPU tensor shape or coefficient differs")
    if not all(torch.isfinite(value).all().item() for value in (chosen, rejected, margin)):
        raise ValueError("nonfinite paired training reward")
    differences = (chosen.float() - rejected.float() - margin.float()).reshape(-1, 2)
    # Maximizing the two BT losses equals training on the worse source format.
    losses = functional.softplus(-differences)
    return losses.max(dim=1).values.mean() + penalty * differences.square().mean()


@contextmanager
def coupled_training(trainer, native_loss, penalty):
    remote_only()
    import torch
    old_collator, old_loss = trainer.PairCollator, trainer.bt_loss

    def loss(chosen, rejected, margin):
        if not torch.is_grad_enabled():
            return native_loss(chosen, rejected, margin)
        return tensor_loss(chosen, rejected, margin, penalty)

    trainer.PairCollator, trainer.bt_loss = grouped_collator(old_collator), loss
    try:
        yield
    finally:
        trainer.PairCollator, trainer.bt_loss = old_collator, old_loss


def gradient_smoke(root, penalty):
    remote_only()
    import torch
    if torch.cuda.is_available():
        raise RuntimeError("paired objective smoke permits public CPU only")
    cases = [(4., -1.), (-1., 4.), (0., 0.), (2., 2.)]
    observed = []
    for first, second in cases:
        chosen = torch.tensor([first, second], dtype=torch.float32, requires_grad=True)
        rejected, margin = torch.zeros(2), torch.zeros(2)
        loss = tensor_loss(chosen, rejected, margin, penalty)
        if abs(loss.item() - scalar_loss(first, second, penalty)) > 2e-6:
            raise ValueError("actual Torch objective differs from scalar contract")
        loss.backward()
        gradients = chosen.grad.detach().tolist()
        if not torch.isfinite(chosen.grad).all().item():
            raise ValueError("nonfinite actual paired gradients")
        if first != second and gradients[0 if first < second else 1] >= 0:
            raise ValueError("weak format has no corrective actual gradient")
        shifted = tensor_loss(chosen.detach() + 7, rejected + 7, margin, penalty)
        if shifted.item() != loss.item():
            raise ValueError("paired objective depends on an absolute reward offset")
        observed.append({"authored_margins": [first, second], "actual_loss": loss.item(), "actual_gradients": gradients})
    refusals = 0
    for chosen in (torch.ones(3), torch.tensor([float("nan"), 0.])):
        try:
            tensor_loss(chosen, torch.zeros_like(chosen), torch.zeros_like(chosen), penalty)
        except ValueError:
            refusals += 1
        else:
            raise ValueError("actual malformed tensor negative control failed")
    report = {"schema_version": 1, "family": FAMILY, "run_id": int(os.environ["GITHUB_RUN_ID"]),
              "commit": os.environ["GITHUB_SHA"], "status": "PAIRED_OBJECTIVE_TORCH_SMOKE_PASSED",
              "cases": observed, "negative_controls": refusals,
              "limits": ["Authored arithmetic/gradient controls, not model inference, quality or reference-engine vectors."]}
    write_json(root / "paired-objective-smoke.json", report)
    return report
