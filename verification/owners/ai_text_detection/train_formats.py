# Optimizer/reward procedure portions: Copyright (c) 2026 telegraphprotocol, MIT.
# Primary trainer 1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd and its full grant are retained in sources.json.
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only Bradley-Terry continuation with weakest-format validation selection.

The optimizer, collator and reward loss follow the pinned Telegraph source.
Our addition selects an actual epoch using every claimed format and label.
"""
import gc
import os
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
    from train_dropout import verify as verify_dropout
    verify_dropout(model, tok, train_rows, hp, trainer, root)
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
    from source_preferences import preference_policy, preference_rows
    supplementary_policy = preference_policy(recipe)
    supplementary_pairs, supplementary_binding = preference_rows(origin_aux.extra_views, train_rows, validation_formats["json"])
    expected_supplementary = json.loads((root / "supplement-preference-contract.json").read_text())
    if supplementary_binding["preference_rows_sha256"] != expected_supplementary["preference_rows_sha256"]:
        raise ValueError("supplementary training preference bytes changed after source/data gate")
    supplementary_by_view = {r["view_id"]: [] for r in supplementary_pairs}
    for row in supplementary_pairs:
        supplementary_by_view[row["view_id"]].append(row)
    if len(supplementary_by_view) != 64 or any(len(v) != 2 for v in supplementary_by_view.values()):
        raise ValueError("complete paired supplementary preferences required")
    from secondary_contract import secondary_preferences
    secondary_pairs, secondary_binding = secondary_preferences(origin_aux.secondary_views, train_rows + supplementary_pairs,
        bare_groups=recipe["secondary_bare_groups"])
    secondary_weights, secondary_weight_policy = weight_policy(secondary_pairs, recipe["bare_loss_weight"])
    secondary_weight_by_id = {r["text_id"]: w / secondary_weight_policy["mean_training_weight"] for r, w in zip(secondary_pairs, secondary_weights)}
    secondary_expected = json.loads((root / "secondary-preference-contract.json").read_text())
    if secondary_binding["preference_rows_sha256"] != secondary_expected["preference_rows_sha256"]:
        raise ValueError("secondary preference bytes changed after data gate")
    for row in secondary_pairs:
        supplementary_by_view.setdefault(row["view_id"], []).append(row)
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
    from margin_contract import margin_policy
    from train_margin import alignment
    aligned_policy = margin_policy(recipe)
    from source_margin import source_margin_policy
    from train_source_margin import consistency
    source_consistency_policy = source_margin_policy(recipe)
    parameters = [p for p in model.parameters() if p.requires_grad]
    parameters += list(origin_aux.head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=hp.lr, weight_decay=hp.weight_decay)
    supplement_groups = {}
    for row in origin_aux.extra_views + origin_aux.secondary_views:
        supplement_groups.setdefault(row["view_id"], []).append(row)
    if len(supplement_groups) != 128 or any(len(v) != 2 for v in supplement_groups.values()):
        raise ValueError("complete independently paired supplementary source phase required")
    supplement_batches = [supplement_groups[k] for k in sorted(supplement_groups) if k.startswith("supplement:")]
    supplement_batches += [supplement_groups[k] for k in sorted(supplement_groups) if k.startswith("secondary:")]
    if len(supplement_batches) != 128:
        raise ValueError("supplementary phase identities changed")
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
        margin_loss_total = 0.0
        source_consistency_total = 0.0
        for i, batch in enumerate(loader):
            chosen = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rejected = trainer._reward(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
            weight = batch["training_weight"] / policy["mean_training_weight"]
            row_losses = []
            pair_values, point_values = [], []
            for j, text_id in enumerate(batch["source_text_ids"]):
                raw_loss, pair_loss, point_loss = loss_components(chosen[j:j+1], rejected[j:j+1], batch["margin"][j:j+1], objective, trainer)
                view = origin_aux.by_original_id[text_id]
                origin_loss, correct, source_logit = origin_aux.loss_with_logit(view)
                source_consistency = consistency(chosen[j:j+1], rejected[j:j+1], source_logit, view["human"], source_consistency_policy)
                row_losses.append(raw_loss * weight + origin_aux.policy["coefficient"] * origin_loss +
                    source_consistency_policy["coefficient"] * source_consistency)
                source_consistency_total += source_consistency.item()
                pair_values.append(pair_loss.item())
                point_values.append(point_loss.item())
                origin_losses += origin_loss.item()
                origin_correct += correct
            margin_loss = alignment(chosen, rejected)
            loss = (coupled_loss(row_losses) + aligned_policy["coefficient"] * margin_loss) / group_accum
            loss.backward()
            losses += loss.item() * group_accum
            unweighted_losses += sum(pair_values) / 2
            pointwise_losses += sum(point_values) / 2
            counterpart_gap += abs(row_losses[0].item() - row_losses[1].item())
            margin_loss_total += margin_loss.item()
            if (i + 1) % group_accum == 0:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
        supplementary_loss, supplementary_correct = 0.0, 0
        supplementary_pair_loss = 0.0
        supplementary_correctness_loss = 0.0
        supplementary_margin_loss = 0.0
        secondary_loss, secondary_correct = 0.0, 0
        extra_source_consistency_total = 0.0
        optimizer.zero_grad(set_to_none=True)
        for i, group in enumerate(supplement_batches):
            preference_group = supplementary_by_view[group[0]["view_id"]]
            preference_batch = basic_collator(preference_group)
            extra_chosen = trainer._reward(model, preference_batch["chosen_input_ids"], preference_batch["chosen_attention_mask"])
            extra_rejected = trainer._reward(model, preference_batch["rejected_input_ids"], preference_batch["rejected_attention_mask"])
            terms = []
            source_losses = {}
            source_logits = {}
            source_roles = {}
            for view in group:
                source_loss, correct, source_logit = origin_aux.loss_with_logit(view)
                supplementary_loss += source_loss.item()
                supplementary_correct += correct
                if i >= 64:
                    secondary_loss += source_loss.item()
                    secondary_correct += correct
                source_losses[view["text_id"]] = source_loss
                source_logits[view["text_id"]] = source_logit
                source_roles[view["text_id"]] = view["human"]
            for j, row in enumerate(preference_group):
                raw_loss, pair_loss, point_loss = loss_components(extra_chosen[j:j+1], extra_rejected[j:j+1],
                    preference_batch["margin"][j:j+1], objective, trainer)
                phase_weight = secondary_weight_by_id[row["text_id"]] if i >= 64 else 1.0
                source_consistency = consistency(extra_chosen[j:j+1], extra_rejected[j:j+1], source_logits[row["text_id"]],
                    source_roles[row["text_id"]], source_consistency_policy)
                terms.append(supplementary_policy["pair_coefficient"] * phase_weight * raw_loss +
                    origin_aux.policy["coefficient"] * source_losses[row["text_id"]] +
                    source_consistency_policy["coefficient"] * source_consistency)
                extra_source_consistency_total += source_consistency.item()
                supplementary_pair_loss += pair_loss.item()
                supplementary_correctness_loss += point_loss.item()
            extra_margin_loss = alignment(extra_chosen, extra_rejected)
            supplementary_margin_loss += extra_margin_loss.item()
            ((coupled_loss(terms) + aligned_policy["coefficient"] * extra_margin_loss) / group_accum).backward()
            if epoch == 0 and i in (0, 64):
                touched_head = [name for name, p in model.named_parameters() if
                    ("classifier" in name or ".score." in name) and p.grad is not None and torch.count_nonzero(p.grad).item()]
                touched_encoder = [name for name, p in model.named_parameters() if
                    "lora_" in name and p.grad is not None and torch.count_nonzero(p.grad).item()]
                if not touched_head or not touched_encoder or any(not torch.isfinite(p.grad).all() for p in parameters if p.grad is not None):
                    raise ValueError("supplementary preferences failed actual reward-head/encoder gradient control")
                control_name = "supplement-preference-gradient-control.json" if i == 0 else "secondary-preference-gradient-control.json"
                dump(root / control_name, {"passed": True,
                    "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
                    "preference_rows_sha256": supplementary_binding["preference_rows_sha256"] if i == 0 else secondary_binding["preference_rows_sha256"],
                    "nonzero_reward_head_gradients": len(touched_head), "nonzero_lora_gradients": len(touched_encoder),
                    "finite_gradients": True, "optimizer_updates_before_control": step,
                    "supplementary_optimizer_updates_before_control": 0, "audit_used": False})
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
                    "supplement_source_only_rows": 256, "supplement_optimizer_steps": 32,
                    "supplement_training_origin_loss": supplementary_loss / 256,
                    "supplement_training_origin_accuracy": supplementary_correct / 256,
                    "supplement_preference_rows": 256, "combined_literal_training_rows": 1008,
                    "supplement_preference_pair_loss": supplementary_pair_loss / 256,
                    "supplement_preference_correctness_loss": supplementary_correctness_loss / 256,
                    "supplement_preference_margin_loss": supplementary_margin_loss / 128,
                    "source_phase_updates": {"original": 94, "first_supplement": 16, "secondary": 16},
                    "secondary_training_origin_loss": secondary_loss / 128,
                    "secondary_training_origin_accuracy": secondary_correct / 128,
                    "secondary_preference_rows_sha256": secondary_binding["preference_rows_sha256"],
                    "secondary_training_weight_policy": secondary_weight_policy,
                    "secondary_new_bare_rows": secondary_binding["new_bare_rows"],
                    "source_margin_consistency_policy": source_consistency_policy,
                    "original_source_margin_loss": source_consistency_total / 752,
                    "extra_source_margin_loss": extra_source_consistency_total / 256,
                    "supplement_preference_policy": supplementary_policy,
                    "supplement_preference_rows_sha256": supplementary_binding["preference_rows_sha256"],
                   "training_margin_alignment_loss": margin_loss_total / len(loader), "margin_alignment": aligned_policy,
                   "lora_training_dropout": hp.lora_dropout,
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
