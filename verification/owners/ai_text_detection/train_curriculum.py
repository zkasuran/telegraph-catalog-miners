# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote format-isolated replay with all previous losses and no new data."""
import json
import os

from common import remote_only, HERE, dump
from curriculum_contract import plan, rounds


def replay(model, tok, origin_aux, rows, weights_by_id, hp, trainer, optimizer, scheduler, root, epoch, original_steps):
    remote_only()
    import torch
    from objective_contract import objective_policy
    from train_objective import loss_components
    from train_coupled import coupled_loss
    from margin_contract import margin_policy
    from train_margin import alignment
    from source_margin import source_margin_policy
    from train_source_margin import consistency
    from verdict_contrast import contrast_policy
    from train_verdict_contrast import loss as contrast
    recipe = json.loads((HERE / "recipe.json").read_text())
    groups, binding = plan(rows, recipe)
    expected = json.loads((HERE / "curriculum-plan.json").read_text())
    if binding != expected:
        raise ValueError("actual curriculum inputs differ from source-frozen plan")
    policy = binding["policy"]
    objective = objective_policy(recipe, rows[:752])
    margin, source, cross = margin_policy(recipe), source_margin_policy(recipe), contrast_policy(recipe)
    parameters = [p for p in model.parameters() if p.requires_grad] + list(origin_aux.head.parameters())
    source_by_id = {r["text_id"]: r for r in origin_aux.extra_views + origin_aux.secondary_views}
    source_by_id.update(origin_aux.by_original_id)
    if any(r["text_id"] not in weights_by_id or r["text_id"] not in source_by_id for group in groups for r in group):
        raise ValueError("curriculum source or phase-normalized weight identity missing")
    collator = trainer.PairCollator(tok, hp.max_length)
    old_rates = [group["lr"] for group in optimizer.param_groups]
    scheduler_before = scheduler.last_epoch
    model.train()
    origin_aux.head.train()
    updates, presentations, total_loss = 0, 0, 0.0
    records = []
    try:
        for group in optimizer.param_groups:
            group["lr"] = hp.lr
        for index, ordered in enumerate(rounds(groups, epoch, policy)):
            optimizer.zero_grad(set_to_none=True)
            for batch_index, group in enumerate(ordered):
                batch = collator(group)
                chosen = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
                rejected = trainer._reward(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
                terms = []
                for side, row in enumerate(group):
                    raw, _, _ = loss_components(chosen[side:side+1], rejected[side:side+1], batch["margin"][side:side+1], objective, trainer)
                    view = source_by_id[row["text_id"]]
                    origin, _, logit = origin_aux.loss_with_logit(view)
                    teacher = consistency(chosen[side:side+1], rejected[side:side+1], logit, view["human"], source)
                    terms.append(weights_by_id[row["text_id"]] * raw + origin_aux.policy["coefficient"] * origin + source["coefficient"] * teacher)
                loss = coupled_loss(terms) + margin["coefficient"] * alignment(chosen, rejected) + cross["coefficient"] * contrast(chosen, rejected)
                if not torch.isfinite(loss).all():
                    raise ValueError("nonfinite curriculum objective")
                (loss / policy["views_per_update"]).backward()
                total_loss += loss.item()
                presentations += 2
                if (batch_index + 1) % policy["views_per_update"] == 0:
                    head = [n for n, p in model.named_parameters() if ("classifier" in n or ".score." in n) and p.grad is not None and torch.count_nonzero(p.grad).item()]
                    encoder = [n for n, p in model.named_parameters() if "lora_" in n and p.grad is not None and torch.count_nonzero(p.grad).item()]
                    if not head or not encoder or any(not torch.isfinite(p.grad).all() for p in parameters if p.grad is not None):
                        raise ValueError("curriculum failed finite production-head/encoder gradient control")
                    norm = torch.nn.utils.clip_grad_norm_(parameters, policy["clip_max_norm"], error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    records.append({"round": index, "update": updates, "bare_view_ids": [g[0]["view_id"] for g in ordered[batch_index-3:batch_index+1]],
                                    "literal_rows": 8, "pre_clip_norm": norm.item(), "learning_rate": hp.lr,
                                    "nonzero_head_gradients": len(head), "nonzero_lora_gradients": len(encoder)})
                    dump(root / "curriculum-progress.json", {"epoch": epoch, "updates": updates, "presentations": presentations,
                         "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"], "audit_used": False})
    finally:
        for group, rate in zip(optimizer.param_groups, old_rates):
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
    if updates != 8 or presentations != 64 or scheduler.last_epoch != scheduler_before or [g["lr"] for g in optimizer.param_groups] != old_rates:
        raise ValueError("curriculum changed fixed update budget or original scheduler state")
    report = {"run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"], "epoch": epoch, "binding": binding,
              "unique_replayed_rows": 16, "training_presentations": presentations, "additional_optimizer_updates": updates,
              "all_optimizer_updates_before_replay": original_steps, "original_scheduler_position": scheduler_before, "mean_training_loss": total_loss / 32,
              "original_scheduler_and_learning_rates_restored": True, "original_losses_and_weights_retained": True,
              "audit_used": False, "validation_replayed": False, "update_controls": records,
              "limits": "Training measurements only. No independent quality or reference-engine result is claimed."}
    path = root / "curriculum-training-report.json"
    previous = json.loads(path.read_text()) if path.exists() else {"epochs": []}
    previous["epochs"].append(report)
    dump(path, previous)
    return report
