# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Development-only relevance curriculum. Neural work is restricted to public CPU CI."""
import gc
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

from data import FORMATS, check_splits, render
from scorer import FAMILY, KIT, BASE, pairs, preserve, remote_only
from source import HERE, sha, write


def validate_development(config):
    expected = {"schema_version": 1, "x-licence": "LicenseRef-zkasuran-SAND-1.0", "family": FAMILY,
                "strategy": "length-balanced-head-warmup-and-grouped-format-rotation-v2",
                "fit_split": "train", "selection_split": "validation", "train_originals": 128,
                "validation_originals": 32, "head_warmup_epochs": 3, "joint_epochs": 12,
                "head_learning_rate": .003, "joint_learning_rate": .0002, "lora_rank": 16,
                "lora_alpha": 32, "max_length": 256, "grad_accum": 16, "seed": 42,
                "loss_weighting": "train-only inverse length-stratum frequency capped at two",
                "selection_metric": "mean_three_format_validation_loss", "tie_rule": "earlier_stage_epoch",
                "retired_audit_sha256": "75701820c853b3f1348ae930271779dfd253deff2b5a9f3230cb6fbae916eb5c",
                "old_audit_scoring_allowed": False, "registration_ready": False,
                "status": "DEVELOPMENT_ONLY_UNMEASURED_V2"}
    if config != expected or any(type(config[key]) is not type(value) for key, value in expected.items()):
        raise ValueError("exact reviewed development contract required; no quality/audit gate override")


def development_splits(directory):
    manifest = json.loads((directory / "freeze.json").read_bytes())
    if manifest["family"] != FAMILY or manifest["audit_used_for_selection"] is not False:
        raise ValueError("original source identity changed")
    splits = {}
    for split, count in (("train", 128), ("validation", 32)):
        raw = (directory / (split + ".json")).read_bytes()
        if sha(raw) != manifest["splits"][split]["sha256"]:
            raise ValueError("development split bytes changed")
        splits[split] = json.loads(raw)
        if len(splits[split]) != count:
            raise ValueError("original development quotas changed")
    check_splits(splits)
    return manifest, splits  # Exposed audit rows are not opened or scored here.


def epoch_pairs(rows, ordinal):
    if type(ordinal) is not int or not 1 <= ordinal <= 15:
        raise ValueError("bounded stage epoch required")
    if len({row["block_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate source original")
    result = [pairs([row], FORMATS[(int(row["block_id"], 16) + ordinal - 1) % 3])[0] for row in rows]
    bucket = lambda row: (len(row["chosen"]) > len(row["rejected"])) - (len(row["chosen"]) < len(row["rejected"]))
    counts = Counter(bucket(row) for row in result)
    for row in result:
        row["loss_weight"] = min(2.0, len(result) / (len(counts) * counts[bucket(row)]))
    return result


def choose(best, ordinal, metrics):
    if set(metrics) != set(FORMATS) or type(ordinal) is not int or not 1 <= ordinal <= 15:
        raise ValueError("all development formats and bounded epoch required")
    losses = [metrics[f]["val_loss"] for f in FORMATS]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in losses):
        raise ValueError("finite validation measurements required")
    value = sum(losses) / 3
    improved = best is None or value < best["val_loss"]
    return ({"stage_epoch": ordinal, "val_loss": value} if improved else best), improved


def validation_metrics(model, tokenizer, trainer, rows, hp):
    remote_only()
    import torch
    from torch.utils.data import DataLoader
    measured = {}
    model.eval()
    with torch.no_grad():
        for f in FORMATS:
            examples = pairs(rows, f)
            value = trainer.evaluate(model, tokenizer, examples, hp)
            diffs = []
            for batch in DataLoader(examples, batch_size=1, shuffle=False, collate_fn=trainer.PairCollator(tokenizer, 256)):
                chosen = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
                rejected = trainer._reward(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
                diffs.extend((chosen - rejected).float().cpu().tolist())
            if len(diffs) != len(rows) or any(not math.isfinite(item) for item in diffs):
                raise ValueError("validation alignment, count or finite-logit failure")
            strata, wins = Counter(), Counter()
            for example, difference in zip(examples, diffs):
                length = len(example["chosen"]) - len(example["rejected"])
                label = "chosen_longer" if length > 0 else "chosen_shorter" if length < 0 else "equal_length"
                strata[label] += 1
                wins[label] += difference > 0
            value.update({"per_label_counts": dict(strata), "per_label": {label: wins[label] / count for label, count in strata.items()},
                          "score_ties": sum(item == 0 for item in diffs), "scope": "validation native logits only, not reference-engine audit"})
            measured[f] = value
    return measured


def develop(root, kit, base, tokenizer_path):
    remote_only()  # Must precede all neural imports and model/config loading.
    config = json.loads((HERE / "development-contract.json").read_bytes())
    validate_development(config)
    manifest, splits = development_splits(HERE / "frozen")
    if json.loads((HERE / "audit-retirement.json").read_bytes())["retired_audit_sha256"] != config["retired_audit_sha256"]:
        raise ValueError("retired audit identity changed")
    identity = json.loads((kit / "source-identity.json").read_bytes())
    if (identity["commit"] != KIT or identity.get("verified_archive_sha256") != "18d7dbe7425e977b576a70146f901a006b8e4abddbfd351276205faa7f287243"
            or sha((kit / "float_runtime/train_adapter.py").read_bytes()) != "5da2919c66c2d0840104628ab88e98a9c8e5e578686f71ca9be4dcd91bd18d71"):
        raise ValueError("reviewed primary trainer source changed")
    if sha(tokenizer_path.read_bytes()) != "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456":
        raise ValueError("released tokenizer identity changed")
    if json.loads((base / "source-identity.json").read_bytes())["revision"] != BASE:
        raise ValueError("base source revision changed")
    sys.path.insert(0, str(kit))
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from float_runtime import train_adapter as trainer
    from tokenizers import Tokenizer
    if torch.cuda.is_available():
        raise RuntimeError("standard public CPU development only")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    hp = trainer.Hyperparams(max_length=256, batch_size=1, grad_accum=16, eval_batch=1,
                            lora_rank=16, lora_alpha=32, epochs=12, seed=42, grad_checkpointing=False)
    seed_pairs = epoch_pairs(splits["train"], 1)
    gate = trainer.gate_dataset(seed_pairs, hp)
    write(root / "upstream-training-gate.json", gate)
    if not gate["ok"]:
        raise ValueError("unmodified primary source dataset gate refused development")
    node = Tokenizer.from_file(str(tokenizer_path))
    node.no_padding()
    node.no_truncation()
    model, tokenizer = trainer.build_model(str(base), hp)
    for rows in splits.values():
        for f in FORMATS:
            for pair in pairs(rows, f):
                for side in ("chosen", "rejected"):
                    text = render(pair["prompt"], pair[side])
                    expected = node.encode(text).ids
                    if len(expected) > 256 or tokenizer(text, truncation=False)["input_ids"] != expected:
                        raise ValueError("complete development object/tokenizer alignment failed")
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    def state_hash():
        digest = __import__("hashlib").sha256()
        for name, parameter in model.named_parameters():
            if name in names:
                digest.update(name.encode())
                digest.update(parameter.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
    initial_hash = state_hash()
    preserve(model, trainer, root / "initial")
    best, selected, selected_hash, history, steps = None, None, None, [], 0
    started = time.monotonic()
    for stage, epochs, rate in (("head_warmup", 3, .003), ("joint_lora", 12, .0002)):
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in names and (stage == "joint_lora" or "lora_" not in name))
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("development stage has no actual trainable head/LoRA parameters")
        optimizer = torch.optim.AdamW(parameters, lr=rate, weight_decay=.01)
        for stage_epoch in range(1, epochs + 1):
            ordinal = len(history) + 1
            training = epoch_pairs(splits["train"], ordinal)
            collator = trainer.PairCollator(tokenizer, 256)
            def collate(batch):
                value = collator(batch)
                value["loss_weight"] = torch.tensor([pair["loss_weight"] for pair in batch], dtype=torch.float32)
                return value
            loader = DataLoader(training, batch_size=1, shuffle=True, collate_fn=collate,
                                generator=torch.Generator(device="cpu").manual_seed(42 + ordinal))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for i, batch in enumerate(loader):
                chosen = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
                rejected = trainer._reward(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
                loss = trainer.bt_loss(chosen, rejected, batch["margin"]) * batch["loss_weight"].mean()
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite development loss")
                (loss / 16).backward()
                if (i + 1) % 16 == 0:
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    steps += 1
                if time.monotonic() - started > 150 * 60:
                    raise TimeoutError("bounded development training reserve")
            with torch.random.fork_rng(devices=[]):
                measured = validation_metrics(model, tokenizer, trainer, splits["validation"], hp)
            best, improved = choose(best, ordinal, measured)
            if improved:
                selected = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if name in names}
                selected_hash = state_hash()
            history.append({"stage": stage, "stage_epoch": stage_epoch, "ordinal": ordinal,
                            "validation": measured, "selected": improved, "audit_scored": False})
            preserve(model, trainer, root / "checkpoint")
            write(root / "development-progress.json", {"history": history, "optimizer_steps": steps, "old_audit_scored": False})
        del optimizer
    if selected is None:
        raise ValueError("validation did not select a frozen development candidate")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in names)
            if name in names:
                parameter.copy_(selected[name])
    if state_hash() != selected_hash:
        raise ValueError("selected development checkpoint restore mismatch")
    preserve(model, trainer, root / "selected")
    report = {"schema_version": 1, "status": "DEVELOPMENT_FROZEN_MODEL_NOT_QUALIFIED", "family": FAMILY,
              "run_id": int(__import__("os").environ["GITHUB_RUN_ID"]), "commit": __import__("os").environ["GITHUB_SHA"],
              "contract_sha256": sha((HERE / "development-contract.json").read_bytes()), "contract": config,
              "selected": best, "history": history, "optimizer_steps": steps, "initial_sha256": initial_hash,
              "selected_trainable_sha256": selected_hash, "fit_split": "train", "selection_split": "validation",
              "train_sha256": manifest["splits"]["train"]["sha256"], "validation_sha256": manifest["splits"]["validation"]["sha256"],
              "development_model_stage_audit_rows_read": False, "old_audit_scored": False, "audit_used_for_selection": False,
              "clean_independent_audit_required": True, "registration_ready": False, "runtime_verified": False,
              "selected_files": {path.name: sha(path.read_bytes()) for path in sorted((root / "selected").iterdir()) if path.is_file()}}
    report["initial_files"] = {path.name: sha(path.read_bytes()) for path in sorted((root / "initial").iterdir()) if path.is_file()}
    write(root / "development-model-report.json", report)
    del model, selected, parameters, loader
    gc.collect()
    return report
