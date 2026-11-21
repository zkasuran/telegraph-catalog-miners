# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote native last-four-layer adaptation after the expanded frozen-head control."""

import math
import re
import time

import data
from development_controls import aggregate, variants

EPOCHS = 2
LAST_LAYERS = 4
LORA_LR = 0.0001
HEAD_LR = 0.00001


def adaptation_parameter(name, layers):
    if type(layers) is not int or layers < LAST_LAYERS:
        raise ValueError("supported layer count required")
    if name in {
        "base_model.model.classifier.modules_to_save.default.weight",
        "base_model.model.classifier.modules_to_save.default.bias",
    }:
        return "head"
    match = re.fullmatch(
        r"base_model\.model\.model\.layers\.(\d+)\.(?:attn\.(?:Wqkv|Wo)|mlp\.(?:Wi|Wo))\.lora_[AB]\.default\.weight",
        name,
    )
    if match and layers - LAST_LAYERS <= int(match[1]) < layers:
        return "lora"
    return None


def select_epoch(records):
    if len(records) != EPOCHS + 1 or [r["epoch"] for r in records] != list(
        range(EPOCHS + 1)
    ):
        raise ValueError("complete predeclared stage coverage required")
    for row in records:
        metric = row["validation"]
        if (
            type(metric["pairs"]) is not int
            or metric["pairs"] < 1
            or type(metric["accuracy"]) not in {int, float}
            or not math.isfinite(metric["accuracy"])
        ):
            raise ValueError("actual complete finite validation required")
        if not 0 <= metric["accuracy"] <= 1:
            raise ValueError("invalid validation accuracy")
    # Prefer the earliest candidate on ties, including the no-adaptation control.
    return max(
        range(len(records)), key=lambda i: (records[i]["validation"]["accuracy"], -i)
    )


def train_adapted_head(
    trainer, base, tok, train, validation, adapter, hp, root, strongest
):
    import remote

    remote.remote_only()
    import torch
    from runtime_head import train_frozen_head

    model, untrained, _, warm = train_frozen_head(
        trainer, base, tok, train, validation, adapter, hp, root, strongest
    )
    frozen = {}
    groups = {"head": [], "lora": []}
    for name, parameter in model.named_parameters():
        group = adaptation_parameter(name, model.config.num_hidden_layers)
        parameter.requires_grad_(group is not None)
        if group:
            groups[group].append(parameter)
        elif ".lora_" in name or "classifier" in name or ".head." in name:
            frozen[name] = parameter.detach().cpu().clone()
    if not groups["head"] or not groups["lora"]:
        raise ValueError("literal supported trainable head and LoRA branches required")
    model.gradient_checkpointing_disable()
    model.disable_input_require_grads()
    torch.manual_seed(hp.seed)
    optimizer = torch.optim.AdamW(
        [
            {"params": groups["head"], "lr": HEAD_LR},
            {"params": groups["lora"], "lr": LORA_LR},
        ],
        weight_decay=hp.weight_decay,
    )
    deadline = time.monotonic() + 240 * 60

    def differences(rows, control=False):
        scores, refused = [], 0
        model.eval()
        with torch.no_grad():
            for row in rows:
                encoded = [
                    tok(
                        data.render(row["prompt"], row[side]),
                        truncation=False,
                        return_tensors="pt",
                    )
                    for side in ("chosen", "rejected")
                ]
                if any(batch["input_ids"].shape[1] > 512 for batch in encoded):
                    if not control:
                        raise ValueError("literal validation over token cap")
                    refused += 1
                    continue
                rewards = [
                    trainer._reward(
                        model, batch["input_ids"], batch["attention_mask"]
                    ).item()
                    for batch in encoded
                ]
                scores.append(rewards[0] - rewards[1])
                if time.monotonic() > deadline:
                    raise ValueError("bounded adaptation runtime exceeded")
        if refused:
            return {
                "pairs_requested": len(rows),
                "pairs_scored": len(scores),
                "over_cap_pairs_refused": refused,
                "complete": False,
                "accuracy": None,
            }
        return aggregate(validation if control else rows, scores)

    records = [{"epoch": 0, "validation": differences(validation), "training": None}]
    snapshots = []

    def snapshot():
        return {
            name: p.detach().cpu().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    snapshots.append(snapshot())
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(
            len(train), generator=torch.Generator().manual_seed(hp.seed + epoch)
        ).tolist()
        loss_sum = 0.0
        optimizer.zero_grad(set_to_none=True)
        for i, index in enumerate(order):
            row = train[index]
            rewards = []
            for side in ("chosen", "rejected"):
                text = data.render(row["prompt"], row[side])
                if text != trainer.render(row["prompt"], row[side]):
                    raise ValueError("literal caller mismatch")
                batch = tok(text, truncation=False, return_tensors="pt")
                if batch["input_ids"].shape[1] > 512:
                    raise ValueError("retained training over token cap")
                rewards.append(
                    trainer._reward(model, batch["input_ids"], batch["attention_mask"])
                )
            loss = trainer.bt_loss(rewards[0], rewards[1], torch.zeros_like(rewards[0]))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite actual pairwise loss")
            loss_sum += loss.item()
            # The last partial group gets its actual count, not a smaller weight.
            group_size = min(
                hp.grad_accum, len(order) - (i // hp.grad_accum) * hp.grad_accum
            )
            (loss / group_size).backward()
            if (i + 1) % hp.grad_accum == 0 or i + 1 == len(order):
                norm = torch.nn.utils.clip_grad_norm_(
                    groups["head"] + groups["lora"], 1.0, error_if_nonfinite=True
                )
                if not torch.isfinite(norm):
                    raise ValueError("nonfinite gradient")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if time.monotonic() > deadline:
                raise ValueError("bounded adaptation runtime exceeded")
        records.append(
            {
                "epoch": epoch,
                "validation": differences(validation),
                "training": {"pairs": len(train), "mean_loss": loss_sum / len(train)},
            }
        )
        snapshots.append(snapshot())
    selected = select_epoch(records)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(snapshots[selected][name])
            elif name in frozen and not torch.equal(
                parameter.detach().cpu(), frozen[name]
            ):
                raise ValueError("unsupported frozen tensor changed")
    actual = {
        name: differences(rows, control=name != "literal")
        for name, rows in variants(validation).items()
    }
    if actual["literal"] != records[selected]["validation"] or actual["prompt_only"][
        "ties"
    ] != len(validation):
        raise ValueError("selected actual ranking or identical-answer control mismatch")
    model.save_pretrained(adapter)
    (adapter / "tokenizer.json").write_bytes((base / "tokenizer.json").read_bytes())
    trainer.save_reward_head(model, adapter)
    result = {
        "schema_version": 1,
        "family": "VENDOR_VERIFY",
        "mechanism": "expanded source groups, ridge warm head and last-four-layer pairwise LoRA adaptation",
        "fit_scope": "training only, source annotations unchanged",
        "train_pairs": len(train),
        "validation_pairs": len(validation),
        "stages": records,
        "selected_index": selected,
        "selection": "unchanged validation accuracy, earliest stage on ties",
        "selected_validation": actual["literal"],
        "development_controls": actual,
        "warm_report_sha256": data.sha(
            (root / "frozen-head-development.json").read_bytes()
        ),
        "frozen_head_validation": warm["selected_validation"],
        "unsupported_frozen_tensors_checked": len(frozen),
        "last_layers": LAST_LAYERS,
        "epochs": EPOCHS,
        "lora_lr": LORA_LR,
        "head_lr": HEAD_LR,
        "audit_evaluated": False,
        "reference_engine_passed": False,
        "registration_ready": False,
        "limits": "Native development only. Frozen head alignment with published foundation and exact Vendor audit still unverified.",
    }
    remote.write_report(root / "adapted-head-development.json", result)
    return model, untrained, trainer.evaluate(model, tok, validation, hp), result
