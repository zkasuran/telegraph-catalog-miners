# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote frozen-feature head fitting with exact input/state controls and no audit."""
import hashlib
import json
import os

from common import remote_only, HERE, dump, validation_gate, validation_selection_key
from headfit_contract import headfit_policy, plan


def objective(chosen, rejected, weights, origin_losses, targets, indexes, recipe):
    remote_only()
    import torch
    from objective_contract import objective_policy
    from margin_contract import margin_policy
    from source_margin import source_margin_policy
    from verdict_contrast import contrast_policy
    from coupled_contract import coupled_policy
    p, m, s, c, g = objective_policy(recipe), margin_policy(recipe), source_margin_policy(recipe), contrast_policy(recipe), coupled_policy(recipe)
    if chosen.ndim != 1 or chosen.shape != rejected.shape or chosen.shape != weights.shape or chosen.shape != origin_losses.shape or chosen.shape != targets.shape:
        raise ValueError("same-shape classifier/source-cache vectors required")
    if any(x.dtype != torch.float32 or x.device.type != "cpu" or not torch.isfinite(x).all() for x in (chosen, rejected, weights, origin_losses, targets)):
        raise ValueError("finite CPU FP32 classifier/source-cache vectors required")
    f = torch.nn.functional
    difference = chosen - rejected
    pair = -f.logsigmoid(difference)
    point = (f.binary_cross_entropy_with_logits(chosen, torch.full_like(chosen, p["chosen_target"]), reduction="none") +
             f.binary_cross_entropy_with_logits(rejected, torch.full_like(rejected, p["rejected_target"]), reduction="none")) / 2
    from origin_contract import origin_policy
    row = (weights * (pair + p["coefficient"] * point) + origin_policy(recipe)["coefficient"] * origin_losses +
           s["coefficient"] * f.smooth_l1_loss(difference, targets, beta=s["beta"], reduction="none"))
    a, b = row[indexes[:, 0]], row[indexes[:, 1]]
    coupled = (a + b) / 2 + g["gap_coefficient"] * (a-b).abs()
    aligned = f.smooth_l1_loss(difference[indexes[:, 0]], difference[indexes[:, 1]], beta=m["beta"], reduction="none")
    cross = f.softplus(rejected[indexes.flip(1)] - chosen[indexes]).mean(dim=1)
    return (coupled + m["coefficient"] * aligned + c["coefficient"] * cross).mean()


def smoke(recipe, trainer, root):
    remote_only()
    import torch
    from objective_contract import objective_policy
    from train_objective import loss_components
    from train_coupled import coupled_loss
    from train_margin import alignment
    from train_verdict_contrast import loss as contrast
    from train_source_margin import consistency
    from source_margin import source_margin_policy
    from margin_contract import margin_policy
    from verdict_contrast import contrast_policy
    from origin_contract import origin_policy
    a = torch.tensor([1., -0.75, 0.5, -1.], requires_grad=True)
    b = torch.tensor([-1., 0.75, -0.5, 1.], requires_grad=True)
    weights = torch.tensor([1., 1., 2., 2.])
    origin = torch.tensor([0.2, 0.3, 0.4, 0.5])
    logits = torch.tensor([0.2, -0.4, 0.3, -0.6])
    roles = torch.tensor([1., -1., 1., -1.])
    s = source_margin_policy(recipe)
    targets = (2 * torch.nn.functional.softplus(roles * logits)).clamp(s["target_min"], s["target_max"])
    indices = torch.tensor([[0, 1], [2, 3]])
    vector = objective(a, b, weights, origin, targets, indices, recipe)
    reference = []
    for group in ((0, 1), (2, 3)):
        terms = []
        for index in group:
            raw, _, _ = loss_components(a[index:index+1], b[index:index+1], torch.zeros(1), objective_policy(recipe), trainer)
            teacher = consistency(a[index:index+1], b[index:index+1], logits[index:index+1], int(roles[index].item() > 0), s)
            terms.append(weights[index]*raw + origin_policy(recipe)["coefficient"]*origin[index] + s["coefficient"]*teacher)
        ids = torch.tensor(group)
        reference.append(coupled_loss(terms) + margin_policy(recipe)["coefficient"]*alignment(a[ids], b[ids]) + contrast_policy(recipe)["coefficient"]*contrast(a[ids], b[ids]))
    direct = torch.stack(reference).mean()
    vector_grad = torch.autograd.grad(vector, (a, b), retain_graph=True)
    reference_grad = torch.autograd.grad(direct, (a, b))
    if not torch.allclose(vector, direct, atol=2e-6, rtol=2e-5) or any(not torch.allclose(x, y, atol=2e-6, rtol=2e-5) for x, y in zip(vector_grad, reference_grad)):
        raise ValueError("head-only vector objective changed retained source losses or gradients")
    refused = []
    for name, bad in (("shape", a[:-1]), ("dtype", a.double()), ("nonfinite", torch.tensor([float("nan"), 1., 1., 1.]))):
        try:
            objective(bad, b, weights, origin, targets, indices, recipe)
        except ValueError:
            refused.append(name)
        else:
            raise ValueError("head-only objective negative control did not fail")
    dump(root / "headfit-objective-controls.json", {"run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
         "passed": True, "vector_loss": vector.item(), "retained_direct_loss": direct.item(), "gradient_match": True, "negative_controls": refused})


def fit(base, train_rows, validation_formats, baselines, adapter, hp, trainer, root, recipe):
    remote_only()
    import torch
    from train_origin import OriginAuxiliary
    from train_dropout import verify as verify_dropout
    from source_preferences import preference_rows
    from secondary_contract import secondary_preferences
    from training_weights import weight_policy
    smoke(recipe, trainer, root)
    model, tok = trainer.build_model(base, hp)
    model.to("cpu")
    verify_dropout(model, tok, train_rows, hp, trainer, root)
    origin = OriginAuxiliary(model, tok, train_rows, validation_formats["json"], recipe, root)
    first, _ = preference_rows(origin.extra_views, train_rows, validation_formats["json"])
    secondary, _ = secondary_preferences(origin.secondary_views, train_rows + first, bare_groups=recipe["secondary_bare_groups"])
    rows = train_rows + first + secondary
    binding = plan(rows, recipe)
    if binding != json.loads((HERE / "headfit-plan.json").read_text()):
        raise ValueError("actual head-fit rows differ from source-frozen training plan")
    policy = headfit_policy(recipe)
    heads = [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and module.out_features == 1 and
             (name.endswith("classifier") or name.endswith("classifier.modules_to_save.default") or name.endswith("score"))]
    if len(heads) != 1:
        raise ValueError("unique actual pinned reward classifier required")
    head_name, head = heads[0]
    head_ids = {id(p) for p in head.parameters()}
    for p in model.parameters():
        p.requires_grad_(id(p) in head_ids)
        p.grad = None
    for p in origin.head.parameters():
        p.requires_grad_(False)
        p.grad = None
    if len([p for p in model.parameters() if p.requires_grad]) != 2 or len(head_ids) != 2:
        raise ValueError("only two actual reward-head tensors may fit")
    model.eval()
    origin.head.eval()

    def frozen_hash():
        digest = hashlib.sha256()
        named = [(n, p) for n, p in model.named_parameters() if id(p) not in head_ids]
        named += [("origin_readout." + n, p) for n, p in origin.head.named_parameters()]
        for name, value in named:
            digest.update(name.encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    frozen_before = frozen_hash()
    captured = []
    identity_checks = 0
    def observe(module, args):
        if len(args) != 1 or args[0].ndim != 2 or args[0].shape != (1, head.in_features) or args[0].dtype != torch.float32:
            raise ValueError("actual classifier-input hook shape/dtype mismatch")
        captured.append(args[0].detach().clone())
    hook = head.register_forward_pre_hook(observe)
    collator = trainer.PairCollator(tok, hp.max_length)
    def cache(pairs):
        nonlocal identity_checks
        values = {"chosen": [], "rejected": []}
        with torch.no_grad():
            for row in pairs:
                batch = collator([row])
                for side in values:
                    captured.clear()
                    native = trainer._reward(model, batch[side + "_input_ids"], batch[side + "_attention_mask"])
                    if len(captured) != 1:
                        raise ValueError("classifier hook did not capture exactly one real input")
                    feature = captured.pop()
                    reconstructed = torch.nn.functional.linear(feature, head.weight, head.bias).squeeze(-1)
                    if not torch.equal(native, reconstructed) or not torch.isfinite(feature).all():
                        raise ValueError("actual classifier-feature score identity failed")
                    identity_checks += 1
                    values[side].append(feature.squeeze(0))
        return {side: torch.stack(features) for side, features in values.items()}
    try:
        training_cache = cache(rows)
        validation_cache = {style: cache(pairs) for style, pairs in validation_formats.items()}
    finally:
        hook.remove()
    cache_bytes = sum(x.numel()*x.element_size() for c in [training_cache, *validation_cache.values()] for x in c.values())
    if cache_bytes > policy["cache_max_bytes"]:
        raise ValueError("classifier feature-cache byte ceiling")
    weights, old_weight_policy = weight_policy(train_rows, recipe["bare_loss_weight"])
    secondary_weights, secondary_policy = weight_policy(secondary, recipe["bare_loss_weight"])
    weights = [w / old_weight_policy["mean_training_weight"] for w in weights] + [1.] * len(first) + [w / secondary_policy["mean_training_weight"] for w in secondary_weights]
    source_by_id = {r["text_id"]: r for r in origin.extra_views + origin.secondary_views}
    source_by_id.update(origin.by_original_id)
    from source_margin import source_margin_policy
    source_policy = source_margin_policy(recipe)
    source_losses, targets = [], []
    with torch.no_grad():
        for row in rows:
            view = source_by_id[row["text_id"]]
            source_loss, _, logit = origin.loss_with_logit(view)
            sign = 1 if view["human"] else -1
            target = torch.clamp(2 * torch.nn.functional.softplus(sign * logit), source_policy["target_min"], source_policy["target_max"])
            source_losses.append(source_loss.item())
            targets.append(target.item())
    weights = torch.tensor(weights, dtype=torch.float32)
    source_losses = torch.tensor(source_losses, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    indexes = torch.tensor(binding["group_indices"], dtype=torch.long)
    optimizer = torch.optim.AdamW(head.parameters(), lr=policy["learning_rate"], weight_decay=hp.weight_decay)
    records = []
    best_key, best_state, best_step = None, None, 0
    def measure():
        metrics = {}
        with torch.no_grad():
            for style, pairs in validation_formats.items():
                cache_ = validation_cache[style]
                differences, losses = [], []
                for index, row in enumerate(pairs):
                    a = head(cache_["chosen"][index:index+1]).squeeze(-1)
                    b = head(cache_["rejected"][index:index+1]).squeeze(-1)
                    differences.append((a-b).item())
                    losses.append(trainer.bt_loss(a, b, torch.zeros_like(a)).item())
                accuracy = sum(v > 0 for v in differences) / len(pairs)
                metrics[style] = {"n_eval": len(pairs), "pairwise_accuracy": round(accuracy, 4), "val_loss": round(sum(losses)/len(pairs), 6),
                    "accuracy_by_label": {label: round(sum(d > 0 for d, r in zip(differences, pairs) if r["label"] == label) /
                        sum(r["label"] == label for r in pairs), 4) for label in ("Human", "AI-generated")}}
        return metrics
    for step in range(policy["steps"] + 1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            a = head(training_cache["chosen"]).squeeze(-1)
            b = head(training_cache["rejected"]).squeeze(-1)
            loss = objective(a, b, weights, source_losses, targets, indexes, recipe)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite head-only objective")
            loss.backward()
            if any(p.grad is not None for p in model.parameters() if id(p) not in head_ids) or any(p.grad is not None for p in origin.head.parameters()):
                raise ValueError("head-only fitting accumulated frozen-state gradients")
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                raise ValueError("head-only gradient identity/finite control failed")
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        if step % policy["validation_interval"] == 0:
            metrics = measure()
            key = validation_selection_key(metrics, baselines)
            selected = best_key is None or key > best_key
            if selected:
                best_key, best_step = key, step
                best_state = {n: p.detach().clone() for n, p in head.state_dict().items()}
            records.append({"step": step, "validation_formats": metrics, "validation_gate": validation_gate(metrics, baselines), "selected": selected,
                            "training_loss": loss.item() if step else None})
            dump(root / "headfit-training-report.json", {"run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
                 "policy": policy, "binding": binding, "classifier_module": head_name, "trainable_tensors": 2, "cache_bytes": cache_bytes,
                 "actual_feature_score_identity_checks": identity_checks, "selected_step": best_step, "records": records,
                 "frozen_state_sha256_before": frozen_before, "audit_used_for_selection": False, "cached_validation_only_for_selection": True})
    head.load_state_dict(best_state, strict=True)
    if frozen_hash() != frozen_before:
        raise ValueError("head-only fitting changed actual frozen encoder/readout bytes")
    final_cached = measure()
    for style, pairs in validation_formats.items():
        actual = trainer.evaluate(model, tok, pairs, hp)
        if actual["pairwise_accuracy"] != final_cached[style]["pairwise_accuracy"] or actual["val_loss"] != final_cached[style]["val_loss"]:
            raise ValueError("cached classifier selection disagrees with actual uncached model validation")
        with torch.no_grad():
            for index, row in enumerate(pairs):
                batch = collator([row])
                for side in ("chosen", "rejected"):
                    native = trainer._reward(model, batch[side + "_input_ids"], batch[side + "_attention_mask"])
                    cached = head(validation_cache[style][side][index:index+1]).squeeze(-1)
                    if not torch.equal(native, cached):
                        raise ValueError("selected head cache/native literal score bits differ")
    model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    trainer.save_reward_head(model, adapter)
    origin.save()
    report = json.loads((root / "headfit-training-report.json").read_text())
    report.update(frozen_state_bits_unchanged=True, no_frozen_parameter_gradients=True, actual_uncached_validation_passed=True,
                  optimizer_updates=policy["steps"], validation_used_for_fit=False)
    dump(root / "headfit-training-report.json", report)
    return model, tok
