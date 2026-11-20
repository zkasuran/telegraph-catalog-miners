# Optimizer/reward procedure portions: Copyright (c) 2026 telegraphprotocol, MIT.
# Primary trainer 1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd and its full grant are retained in sources.json.
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only Bradley-Terry continuation with weakest-format validation selection.

The optimizer, collator and reward loss follow the pinned Telegraph source.
Our addition selects an actual epoch using every claimed format and label.
"""
import gc
import random
import time

from common import remote_only, dump, validation_gate, validation_selection_key
from training_weights import weight_policy
from objective_contract import objective_policy
from coupled_contract import coupled_policy, source_groups


def fit(base, train_rows, validation_formats, baselines, adapter, hp, trainer, root):
    remote_only()  # No neural imports before the controller-owned CPU guard.
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    model, tok = trainer.build_model(base, hp)
    device, dtype = trainer.pick_device()
    if device.type != "cpu" or dtype != torch.float32:
        raise ValueError("standard public CPU FP32 training required")
    model.to(device)
    from common import HERE
    import json
    recipe = json.loads((HERE / "recipe.json").read_text())
    coupled = coupled_policy(recipe)
    groups = source_groups(train_rows)
    weights, policy = weight_policy(train_rows, recipe.get("bare_loss_weight", 1))
    objective = objective_policy(recipe, train_rows)
    from train_objective import loss_components
    dump(root / "training-objective-report.json", objective)
    from train_origin import OriginAuxiliary
    origin_aux = OriginAuxiliary(model, tok, train_rows, validation_formats["json"], recipe, root)
    if hp.batch_size != 1:
        raise ValueError("counterpart training is bounded to one source-view batch")
    dump(root / "training-weight-report.json", policy)
    basic_collator = trainer.PairCollator(tok, hp.max_length)
    def collate(wrapped):
        if len(wrapped) != 1 or len(wrapped[0]) != 2:
            raise ValueError("bounded paired source-view batch required")
        rows = wrapped[0]
        result = basic_collator(rows)
        result["training_weight"] = float(recipe.get("bare_loss_weight", 1) if rows[0]["style"] == "bare_label" else 1)
        result["source_text_ids"] = [r["text_id"] for r in rows]
        return result
    loader = DataLoader(groups, batch_size=1, shuffle=True,
                        collate_fn=collate)
    if hp.grad_accum != coupled["row_accumulation"] or hp.grad_accum % 2 or len(groups) % (hp.grad_accum // 2):
        raise ValueError("coupling must preserve complete eight-row optimizer updates")
    group_accum = hp.grad_accum // 2
    from train_coupled import coupled_loss
    parameters = [p for p in model.parameters() if p.requires_grad]
    parameters += list(origin_aux.head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=hp.lr, weight_decay=hp.weight_decay)
    supplement_groups = {}
    for row in origin_aux.extra_views:
        supplement_groups.setdefault(row["view_id"], []).append(row)
    if len(supplement_groups) != 64 or any(len(v) != 2 for v in supplement_groups.values()):
        raise ValueError("complete independently paired supplementary source phase required")
    supplement_batches = [supplement_groups[k] for k in sorted(supplement_groups)]
    total_steps = (len(loader) // group_accum + len(supplement_batches) // group_accum) * hp.epochs
    warm = int(total_steps * 0.06)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:
        min(1.0, s / max(1, warm)) if s < warm else max(0.0, (total_steps - s) / max(1, total_steps - warm)))
    best, best_epoch, stale, step = None, -1, 0, 0
    epochs = []
    started = time.monotonic()
    def measure():
        metrics = {}
        for style, rows in validation_formats.items():
            result = trainer.evaluate(model, tok, rows, hp)
            result["accuracy_by_label"] = {label: trainer.evaluate(model, tok, [r for r in rows if r["label"] == label], hp)["pairwise_accuracy"]
                                            for label in ("Human", "AI-generated")}
            metrics[style] = result
        return metrics
    # Preserve the actually measured starting checkpoint if all later updates
    # worsen validation. This is not a fabricated epoch or a synthetic metric.
    starting = measure()
    best = validation_selection_key(starting, baselines)
    model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    trainer.save_reward_head(model, adapter)
    origin_aux.save()
    epochs.append({"epoch": -1, "optimizer_steps": 0, "training_loss": None, "validation_formats": starting,
                   "validation_gate": validation_gate(starting, baselines), "selection_key": list(best), "selected": True,
                   "role": "Actual measured source-fitted starting checkpoint, before new fitting."})
    for epoch in range(hp.epochs):
        model.train()
        origin_aux.head.train()
        optimizer.zero_grad(set_to_none=True)
        losses = 0.0
        unweighted_losses = 0.0
        pointwise_losses = 0.0
        origin_losses, origin_correct = 0.0, 0
        counterpart_gap = 0.0
        for i, batch in enumerate(loader):
            chosen = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rejected = trainer._reward(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
            weight = batch["training_weight"] / policy["mean_training_weight"]
            row_losses = []
            pair_values, point_values = [], []
            for j, text_id in enumerate(batch["source_text_ids"]):
                raw_loss, pair_loss, point_loss = loss_components(chosen[j:j+1], rejected[j:j+1], batch["margin"][j:j+1], objective, trainer)
                origin_loss, correct = origin_aux.loss(origin_aux.by_original_id[text_id])
                row_losses.append(raw_loss * weight + origin_aux.policy["coefficient"] * origin_loss)
                pair_values.append(pair_loss.item())
                point_values.append(point_loss.item())
                origin_losses += origin_loss.item()
                origin_correct += correct
            loss = coupled_loss(row_losses) / group_accum
            loss.backward()
            losses += loss.item() * group_accum
            unweighted_losses += sum(pair_values) / 2
            pointwise_losses += sum(point_values) / 2
            counterpart_gap += abs(row_losses[0].item() - row_losses[1].item())
            if (i + 1) % group_accum == 0:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
        supplementary_loss, supplementary_correct = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for i, group in enumerate(supplement_batches):
            terms = []
            for view in group:
                source_loss, correct = origin_aux.loss(view)
                supplementary_loss += source_loss.item()
                supplementary_correct += correct
                terms.append(origin_aux.policy["coefficient"] * source_loss)
            (coupled_loss(terms) / group_accum).backward()
            if (i + 1) % group_accum == 0:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
        metrics = measure()
        origin_diagnostic = origin_aux.measure(epoch, origin_losses, origin_correct, 2 * len(loader))
        key = validation_selection_key(metrics, baselines)
        improved = best is None or key > best
        if improved:
            best, best_epoch, stale = key, epoch, 0
            model.save_pretrained(adapter)
            tok.save_pretrained(adapter)
            trainer.save_reward_head(model, adapter)
            origin_aux.save()
        else:
            stale += 1
        record = {"epoch": epoch, "optimizer_steps": step, "training_loss": losses / max(1, len(loader)),
                   "unweighted_training_pair_loss": unweighted_losses / max(1, len(loader)), "training_weight_policy": policy,
                   "unweighted_training_correctness_loss": pointwise_losses / max(1, len(loader)), "training_objective": objective,
                   "origin_auxiliary_diagnostic": origin_diagnostic,
                   "coupled_objective": coupled, "training_counterpart_gap": counterpart_gap / len(loader),
                   "literal_training_rows": 2 * len(loader), "source_view_batches": len(loader),
                   "literal_batch_size": 2, "literal_rows_per_optimizer_update": hp.grad_accum,
                   "supplement_source_only_rows": 128, "supplement_optimizer_steps": 16,
                   "supplement_training_origin_loss": supplementary_loss / 128,
                   "supplement_training_origin_accuracy": supplementary_correct / 128,
                  "validation_formats": metrics, "validation_gate": validation_gate(metrics, baselines),
                  "selection_key": list(key), "selected": improved}
        epochs.append(record)
        dump(root / "format-training-report.json", {"epochs": epochs, "selected_epoch": best_epoch,
             "selection": "Actual validation only; gate pass and weakest format/label margin before mean loss.",
             "audit_used_for_selection": False, "elapsed_seconds": time.monotonic() - started})
        print(record, flush=True)
        if stale > hp.es_patience or record["training_loss"] < hp.loss_floor:
            break
    del optimizer, parameters
    origin_aux.assert_excluded()
    gc.collect()
    model.load_adapter(str(adapter), adapter_name="default", is_trainable=False)
    model.set_adapter("default")
    model.to(device)
    return model, tok
