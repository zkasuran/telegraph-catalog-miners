# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote joint format replay. Original source phases and quality gates stay."""
import json
import os

from common import remote_only, HERE, dump
from transfer_contract import plan, schedule


def replay(model, tok, origin_aux, rows, hp, trainer, optimizer, scheduler, root, epoch, previous_steps):
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
    if binding != json.loads((HERE / "transfer-plan.json").read_text()):
        raise ValueError("actual format-transfer inputs differ from frozen training plan")
    policy = binding["policy"]
    objective = objective_policy(recipe, rows[:752])
    margin, source, cross = margin_policy(recipe), source_margin_policy(recipe), contrast_policy(recipe)
    source_by_id = {r["text_id"]: r for r in origin_aux.extra_views + origin_aux.secondary_views}
    source_by_id.update(origin_aux.by_original_id)
    parameters = [p for p in model.parameters() if p.requires_grad] + list(origin_aux.head.parameters())
    collator = trainer.PairCollator(tok, hp.max_length)
    old_rates = [g["lr"] for g in optimizer.param_groups]
    scheduler_before = scheduler.last_epoch
    model.train()
    origin_aux.head.train()
    updates, presentations, total_loss, controls = 0, 0, 0., []
    try:
        for group in optimizer.param_groups:
            group["lr"] = hp.lr
        for block in schedule(groups, epoch, policy):
            optimizer.zero_grad(set_to_none=True)
            for group in block["groups"]:
                batch = collator(group)
                chosen = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
                rejected = trainer._reward(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
                terms = []
                for index, row in enumerate(group):
                    view = source_by_id[row["text_id"]]
                    raw, _, _ = loss_components(chosen[index:index+1], rejected[index:index+1], batch["margin"][index:index+1], objective, trainer)
                    origin, _, logit = origin_aux.loss_with_logit(view)
                    teacher = consistency(chosen[index:index+1], rejected[index:index+1], logit, view["human"], source)
                    terms.append(policy["replay_pair_weight"] * raw + origin_aux.policy["coefficient"] * origin + source["coefficient"] * teacher)
                loss = coupled_loss(terms) + margin["coefficient"] * alignment(chosen, rejected) + cross["coefficient"] * contrast(chosen, rejected)
                if not torch.isfinite(loss).all():
                    raise ValueError("nonfinite format-transfer objective")
                (loss / 4).backward()
                total_loss += loss.item()
                presentations += 2
            head = [n for n, p in model.named_parameters() if ("classifier" in n or ".score." in n) and p.grad is not None and torch.count_nonzero(p.grad).item()]
            encoder = [n for n, p in model.named_parameters() if "lora_" in n and p.grad is not None and torch.count_nonzero(p.grad).item()]
            if not head or not encoder or any(not torch.isfinite(p.grad).all() for p in parameters if p.grad is not None):
                raise ValueError("format-transfer production-head/encoder gradient control failed")
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            controls.append({"round": block["round"], "update": updates,
                             "view_ids": [g[0]["view_id"] for g in block["groups"]],
                             "formats": [g[0]["style"] for g in block["groups"]], "literal_rows": 8,
                             "nonzero_head_gradients": len(head), "nonzero_lora_gradients": len(encoder),
                             "pre_clip_norm": norm.item(), "learning_rate": hp.lr})
            dump(root / "transfer-progress.json", {"run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
                 "epoch": epoch, "updates": updates, "presentations": presentations, "audit_used": False})
    finally:
        for group, rate in zip(optimizer.param_groups, old_rates):
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
    if updates != 24 or presentations != 192 or scheduler.last_epoch != scheduler_before or [g["lr"] for g in optimizer.param_groups] != old_rates:
        raise ValueError("format transfer changed its fixed budget or original scheduler")
    report = {"run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"], "epoch": epoch, "binding": binding,
              "unique_replayed_rows": 48, "training_presentations": presentations, "additional_optimizer_updates": updates,
              "all_optimizer_updates_before_replay": previous_steps, "original_scheduler_position": scheduler_before,
              "original_scheduler_and_learning_rates_restored": True, "original_source_losses_and_weights_retained": True,
              "replay_pair_weight": 1., "mean_training_loss": total_loss / 96, "update_controls": controls,
              "audit_used": False, "validation_replayed": False,
              "limits": "Training execution only. Uniform weights apply only to joint replay, not original source phases or quality gates."}
    path = root / "transfer-training-report.json"
    previous = json.loads(path.read_text()) if path.exists() else {"epochs": []}
    previous["epochs"].append(report)
    dump(path, previous)
    return report
