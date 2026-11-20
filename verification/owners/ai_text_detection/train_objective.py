# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only stable correctness auxiliary on existing literal pair logits.

[TORCH-BCE-2.15] DATA-SOURCES.md cites the primary API and source formula.
No sigmoid, score rescaling or format transformation is added to the executor.
"""
from common import remote_only, dump
from objective_contract import scalar_losses


def loss_components(chosen, rejected, margin, policy, trainer):
    remote_only()
    import torch
    if (chosen.shape != rejected.shape or chosen.shape != margin.shape or
            chosen.ndim != 1 or chosen.numel() != 1 or
            any(x.device.type != "cpu" or x.dtype != torch.float32 for x in (chosen, rejected, margin)) or
            any(not torch.isfinite(x).all() for x in (chosen, rejected, margin)) or
            torch.count_nonzero(margin).item()):
        raise ValueError("finite same-shape batch-one CPU FP32 zero-margin logits required")
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    pair = trainer.bt_loss(chosen, rejected, margin)
    point = (criterion(chosen, torch.full_like(chosen, policy["chosen_target"])) +
             criterion(rejected, torch.full_like(rejected, policy["rejected_target"]))) / 2
    total = pair + policy["coefficient"] * point
    if not torch.isfinite(total).all():
        raise ValueError("nonfinite training objective")
    return total, pair, point


def smoke(policy, trainer, root):
    remote_only()
    import torch
    controls = []
    for c, r in ((1.0, -1.0), (-1.0, 1.0), (0.0, 0.0), (1000.0, -1000.0), (-1000.0, 1000.0)):
        chosen = torch.tensor([c], dtype=torch.float32, requires_grad=True)
        rejected = torch.tensor([r], dtype=torch.float32, requires_grad=True)
        margin = torch.zeros_like(chosen)
        total, pair, point = loss_components(chosen, rejected, margin, policy, trainer)
        expected = scalar_losses(c, r, policy)
        for observed, reference in zip((pair, point, total), expected):
            if abs(observed.item() - reference) > 2e-4:
                raise ValueError("primary tensor loss disagrees with scalar formula control")
        total.backward()
        if not torch.isfinite(chosen.grad).all() or not torch.isfinite(rejected.grad).all():
            raise ValueError("nonfinite actual objective gradient")
        if c == r == 0 and not (chosen.grad.item() < 0 < rejected.grad.item()):
            raise ValueError("objective gradient reversed the preference sides")
        controls.append({"chosen": c, "rejected": r, "pair_loss": pair.item(),
                         "pointwise_loss": point.item(), "total_loss": total.item(),
                         "chosen_gradient": chosen.grad.item(), "rejected_gradient": rejected.grad.item()})
    refused = []
    for label, c, r, m in (
            ("shape", torch.zeros(1), torch.zeros(2), torch.zeros(1)),
            ("nonfinite", torch.tensor([float("nan")]), torch.zeros(1), torch.zeros(1)),
            ("margin", torch.zeros(1), torch.zeros(1), torch.ones(1)),
            ("dtype", torch.zeros(1, dtype=torch.float64), torch.zeros(1), torch.zeros(1))):
        try:
            loss_components(c, r, m, policy, trainer)
        except ValueError:
            refused.append(label)
        else:
            raise ValueError("objective failure control did not refuse " + label)
    dump(root / "objective-tensor-controls.json", {"passed": True, "actual_tensor_controls": controls,
          "negative_controls": refused, "model_loaded_for_controls": False,
          "scope": "Actual pinned Torch loss/gradient only, not model quality or node runtime."})
