# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote exact-loss gradient measurements without training or audit inference."""
import importlib.metadata
import json
import os
import platform
import time

from common import remote_only, dump, HERE, sha
from gradient_contract import diagnostic_policy, select_views, geometry


def counterpart_gap(a, b):
    remote_only()
    import torch
    return 0.25 * torch.abs(a - b)


def smoke(root):
    remote_only()
    import torch
    from train_coupled import coupled_loss
    controls = []
    for values, expected in (((1., 3.), (0.25, 0.75)), ((3., 1.), (0.75, 0.25)), ((2., 2.), (0.5, 0.5))):
        a, b = [torch.tensor(v, dtype=torch.float32, requires_grad=True) for v in values]
        exact = coupled_loss([a, b])
        decomposed = (a + b) / 2 + counterpart_gap(a, b)
        actual = torch.autograd.grad(exact, (a, b), retain_graph=True)
        separate = torch.autograd.grad(decomposed, (a, b), retain_graph=True)
        if not torch.equal(exact, decomposed) or any(x.item() != e or not torch.equal(x, y) for x, y, e in zip(actual, separate, expected)):
            raise ValueError("direct diagnostic gap changed original coupled value/gradient")
        controls.append({"inputs": list(values), "gradients": [v.item() for v in actual]})
    a, b = [torch.tensor(v, dtype=torch.float32, requires_grad=True) for v in (1., 3.)]
    actual = torch.autograd.grad(coupled_loss([a, b]), (a, b))
    omitted = torch.autograd.grad((a + b) / 2, (a, b))
    if all(torch.allclose(x, y, atol=2e-6, rtol=2e-5) for x, y in zip(actual, omitted)):
        raise ValueError("omitted component negative control did not fail")
    # A primary-source numerical control, not the cause of the model failure.
    a, b = [torch.tensor(v, dtype=torch.float32) for v in (2**26, 2**26 + 8)]
    direct = counterpart_gap(a, b)
    subtractive = coupled_loss([a, b]) - (a + b) / 2
    if direct.item() != 2 or subtractive.item() != 0:
        raise ValueError("FP32 subtractive gap cancellation control changed")
    dump(root / "gradient-algebra-controls.json", {"run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
         "passed": True, "actual_controls": controls, "omitted_component_refused": True,
         "fp32_cancellation_fixture": {"direct_gap": direct.item(), "subtractive_gap": subtractive.item()},
         "original_gradient_tolerances": {"atol": 2e-6, "rtol": 2e-5},
         "limits": "Actual small tensor controls only. No model-quality or failed-model cause is proved."})


def probe(model, tok, original, first, secondary, origin_aux, hp, trainer, root):
    remote_only()
    import torch
    from peft import get_peft_model_state_dict
    from training_weights import weight_policy
    from objective_contract import objective_policy
    from train_objective import loss_components
    from train_coupled import coupled_loss
    from margin_contract import margin_policy
    from train_margin import alignment
    from source_margin import source_margin_policy
    from train_source_margin import consistency
    from verdict_contrast import contrast_policy
    from train_verdict_contrast import loss as contrast
    from secondary_contract import expanded_source_bundle

    smoke(root)
    recipe = json.loads((HERE / "recipe.json").read_text())
    policy = diagnostic_policy(recipe)
    rows = original + first + secondary
    views = select_views(rows, policy)
    plan = json.loads((HERE / "gradient-diagnostic-plan.json").read_text())
    if (plan["view_ids"] != [g[0]["view_id"] for g in views] or
            plan["training_rows_sha256"] != sha(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()) or
            plan["selected_rows_sha256"] != sha(json.dumps(views, sort_keys=True, ensure_ascii=False).encode())):
        raise ValueError("actual diagnostic inputs differ from the source-frozen training plan")
    objective = objective_policy(recipe, original)
    aligned, source_policy, cross = margin_policy(recipe), source_margin_policy(recipe), contrast_policy(recipe)
    _, original_weights = weight_policy(original, recipe["bare_loss_weight"])
    extra_weights, secondary_weights = weight_policy(secondary, recipe["bare_loss_weight"])
    secondary_weight = {r["text_id"]: w / secondary_weights["mean_training_weight"] for r, w in zip(secondary, extra_weights)}
    original_ids = {r["text_id"] for r in original}
    extra_sources = {r["text_id"]: r for r in origin_aux.extra_views + origin_aux.secondary_views}
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    named += [("origin_readout." + n, p) for n, p in origin_aux.head.named_parameters()]
    parameters = [p for _, p in named]
    subsets = {"shared_encoder": [i for i, (n, _) in enumerate(named) if "lora_" in n],
               "reward_head": [i for i, (n, _) in enumerate(named) if "classifier" in n or ".score." in n],
               "origin_readout": [i for i, (n, _) in enumerate(named) if n.startswith("origin_readout.")]}
    if len(subsets["shared_encoder"]) != 176 or len(subsets["reward_head"]) != 2 or len(subsets["origin_readout"]) != 2:
        raise ValueError("actual diagnostic trainable parameter identities changed")
    before = {n: p.detach().clone() for n, p in get_peft_model_state_dict(model, save_embedding_layers=False).items()}
    before_head = {n: p.detach().clone() for n, p in origin_aux.head.state_dict().items()}
    model.train()
    origin_aux.head.train()
    model.zero_grad(set_to_none=True)
    origin_aux.head.zero_grad(set_to_none=True)
    identity = {"family": "AI_TEXT_DETECTION", "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
                "source_manifest_sha256": expanded_source_bundle()[0], "checkpoint_sha256": recipe["checkpoint_sha256"],
                "readout_checkpoint_sha256": recipe["origin_checkpoint_sha256"],
                "data_sha256": {name: sha((root / name).read_bytes()) for name in ("development-generation.jsonl", "augmentation-generation.jsonl", "additional-augmentation-generation.jsonl", "supplement-generation.jsonl", "secondary-generation.jsonl")},
                "foundation_sha256": sha((root / "foundation/foundation_lora_fp32.onnx").read_bytes()),
                "engine_sha256": sha((root / "foundation/foundation_engine.wasm").read_bytes()),
                "tokenizer_sha256": sha((root / "foundation/tokenizer.json").read_bytes()),
                "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                            "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "numpy", "safetensors")}},
                "package": "No exported candidate package; diagnostic-only execution.", "node_verification": "not_run"}
    report = {**identity, "status": "GRADIENT_DIAGNOSTIC_IN_PROGRESS", "policy": policy, "views": [],
              "optimizer_updates": 0, "validation_used": False, "audit_used": False, "registration_eligible": False}
    started = time.monotonic()
    collator = trainer.PairCollator(tok, hp.max_length)
    report["fixed_training_view_ids"] = [group[0]["view_id"] for group in views]
    report["fitted_models_or_released_packages"] = 0
    dump(root / "gradient-interaction-report.json", report)

    def stats(left, right, indexes):
        ls = rs = dot = 0.0
        for index in indexes:
            a, b = left[index], right[index]
            if a is not None:
                ls += a.double().square().sum().item()
            if b is not None:
                rs += b.double().square().sum().item()
            if a is not None and b is not None:
                dot += (a.double() * b.double()).sum().item()
        return geometry(ls, rs, dot)

    for group in views:
        if time.monotonic() - started > policy["time_limit_seconds"]:
            raise ValueError("gradient diagnostic time ceiling")
        batch = collator(group)
        chosen = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
        rejected = trainer._reward(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
        verdict_terms, origin_terms, teacher_terms = [], [], []
        for i, row in enumerate(group):
            total, _, _ = loss_components(chosen[i:i+1], rejected[i:i+1], batch["margin"][i:i+1], objective, trainer)
            if row["text_id"] in original_ids:
                view = origin_aux.by_original_id[row["text_id"]]
                weight = (recipe["bare_loss_weight"] if row["style"] == "bare_label" else 1) / original_weights["mean_training_weight"]
            else:
                view = extra_sources[row["text_id"]]
                weight = secondary_weight.get(row["text_id"], 1.0)
            origin, _, logit = origin_aux.loss_with_logit(view)
            verdict_terms.append(total * weight)
            origin_terms.append(origin * origin_aux.policy["coefficient"])
            teacher_terms.append(consistency(chosen[i:i+1], rejected[i:i+1], logit, view["human"], source_policy) * source_policy["coefficient"])
        row_totals = [verdict_terms[i] + origin_terms[i] + teacher_terms[i] for i in range(2)]
        components = {"verdict": torch.stack(verdict_terms).mean(), "origin": torch.stack(origin_terms).mean(),
                      "teacher_consistency": torch.stack(teacher_terms).mean(),
                       "counterpart_gap": counterpart_gap(row_totals[0], row_totals[1]),
                      "paired_margin": aligned["coefficient"] * alignment(chosen, rejected),
                      "same_verdict": cross["coefficient"] * contrast(chosen, rejected)}
        exact = coupled_loss(row_totals) + components["paired_margin"] + components["same_verdict"]
        decomposed = sum(components.values())
        if not torch.allclose(exact, decomposed, atol=2e-6, rtol=2e-5):
            raise ValueError("diagnostic changed the exact current objective value")
        gradients = {}
        for name, value in components.items():
            gradients[name] = torch.autograd.grad(value, parameters, retain_graph=True, create_graph=False, allow_unused=True)
            if any(g is not None and not torch.isfinite(g).all() for g in gradients[name]):
                raise ValueError("nonfinite diagnostic component gradients")
        total_gradient = torch.autograd.grad(exact, parameters, create_graph=False, allow_unused=True)
        max_error = 0.0
        for index, parameter in enumerate(parameters):
            combined = torch.zeros_like(parameter)
            for values in gradients.values():
                if values[index] is not None:
                    combined += values[index]
            actual = total_gradient[index] if total_gradient[index] is not None else torch.zeros_like(parameter)
            max_error = max(max_error, (combined - actual).abs().max().item())
            if not torch.allclose(combined, actual, atol=2e-6, rtol=2e-5):
                if not torch.isfinite(actual).all():
                    report.update(status="GRADIENT_DIAGNOSTIC_FAILED", failure={"view_id": group[0]["view_id"],
                        "tensor_name": named[index][0], "reason": "nonfinite actual total gradient"})
                else:
                    bad = (combined - actual).abs() > (2e-6 + 2e-5 * actual.abs())
                    locations = bad.flatten().nonzero().flatten()[:8].tolist()
                    report.update(status="GRADIENT_DIAGNOSTIC_FAILED", failure={"view_id": group[0]["view_id"],
                        "group_id": group[0]["group_id"], "format": group[0]["style"], "tensor_name": named[index][0],
                        "reason": "component gradient sum disagreed with actual full objective", "atol": 2e-6, "rtol": 2e-5,
                        "maximum_absolute_error": (combined - actual).abs().max().item(), "mismatched_elements": bad.sum().item(),
                        "samples": [{"flat_index": i, "actual": actual.flatten()[i].item(), "component_sum": combined.flatten()[i].item(),
                            "component_gradients": {n: (v[index].flatten()[i].item() if v[index] is not None else 0.) for n, v in gradients.items()}}
                            for i in locations]})
                dump(root / "gradient-interaction-report.json", report)
                raise ValueError("component gradient sum disagreed with actual full objective")
        measured = {"group_id": group[0]["group_id"], "view_id": group[0]["view_id"], "format": group[0]["style"],
                    "text_ids": [r["text_id"] for r in group], "exact_loss": exact.item(),
                    "component_losses": {n: v.item() for n, v in components.items()},
                    "component_sum_gradient_passed": True, "maximum_component_gradient_error": max_error,
                    "gradients_vs_verdict": {n: {subset: stats(gradients["verdict"], grad, indexes) for subset, indexes in subsets.items()}
                                             for n, grad in gradients.items()}}
        report["views"].append(measured)
        dump(root / "gradient-interaction-report.json", report)
        del gradients, total_gradient, components, exact, decomposed, batch, chosen, rejected
    after = get_peft_model_state_dict(model, save_embedding_layers=False)
    if any(not torch.equal(before[n], p) for n, p in after.items()) or any(not torch.equal(before_head[n], p) for n, p in origin_aux.head.state_dict().items()):
        raise ValueError("diagnostic modified actual selected learned state")
    if any(p.grad is not None for p in parameters):
        raise ValueError("autograd.grad unexpectedly accumulated parameter gradients")
    report.update(status="GRADIENT_DIAGNOSTIC_COMPLETE", learned_state_bits_unchanged=True,
                  no_accumulated_parameter_gradients=True, elapsed_seconds=time.monotonic() - started)
    dump(root / "gradient-interaction-report.json", report)
