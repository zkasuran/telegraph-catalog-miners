# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Matched remote-only batching controls on the original complete training task."""
import argparse
import gc
import hashlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path

from data import HERE, digest, prepare, read_splits, write_json
from diagnostic import training_controls
from gates import load_trainer
from licensing import licence_metadata
from metrics import pair_metrics
from preserve import preserve
from reference import prepare_kit, prepare_reference
from runtime_check import model_remote_only
from training_policy import isolated_evaluation


def policy():
    value = json.loads((HERE / "reciprocal-policy.json").read_text())
    expected = {"SPDX-License-Identifier": "LicenseRef-zkasuran-SAND-1.0",
                "scope": "training-only matched batching control, not candidate or independent audit",
                "split": "train", "training_groups": 4, "training_pairs": 8, "batch_size": 2,
                "optimizer_steps": 512, "learning_rate": .0001, "lora_rank": 8, "lora_alpha": 32,
                "lora_dropout": 0., "seed": 42, "report_interval": 64,
                "controls": ["stochastic_batch", "reciprocal_batch"], "validation_scored": False,
                "audit_scored": False, "registration_allowed": False}
    if value != expected:
        raise ValueError("closed matched batching policy required")
    return value


def selected_rows(manifest, train):
    frozen = json.loads((HERE / "data/manifest.json").read_text())
    original = [r for g in frozen["splits"]["train"] for r in g["pairs"]]
    if manifest != frozen or train != original:
        raise ValueError("batch controls accept only the unchanged frozen training split")
    return training_controls(manifest, train)["original_two_reference"]


def schedules(rows):
    config = policy()
    if len(rows) != 8 or len({r["group_id"] for r in rows}) != 4:
        raise ValueError("four complete reciprocal groups required")
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["group_id"], []).append(index)
    for group in groups.values():
        if len(group) != 2:
            raise ValueError("incomplete reciprocal batch")
        one, two = (rows[i] for i in group)
        if ({one["target"], two["target"]} != {"A", "B"}
                or (one["chosen"], one["rejected"]) != (two["rejected"], two["chosen"])
                or (one["source_id"], one["donor_id"]) != (two["donor_id"], two["source_id"])):
            raise ValueError("batch must contain exactly reciprocal source preferences")
    result = {}
    for name in config["controls"]:
        randomizer = random.Random(config["seed"])
        batches = []
        while len(batches) < config["optimizer_steps"]:
            if name == "reciprocal_batch":
                order = list(groups.values())
                randomizer.shuffle(order)
                batches.extend(list(group) for group in order)
            else:
                order = list(range(len(rows)))
                randomizer.shuffle(order)
                batches.extend([order[i], order[i + 1]] for i in range(0, len(rows), 2))
        result[name] = batches
    return result


def source_check():
    manifest = json.loads((HERE / "data/manifest.json").read_text())
    rows = selected_rows(manifest, [r for g in manifest["splits"]["train"] for r in g["pairs"]])
    plans = schedules(rows)
    return {"reciprocal_source": "PASS", "training_pairs": len(rows), "training_groups": 4,
            "optimizer_steps_per_control": {n: len(p) for n, p in plans.items()},
            "original_complete_prompts": True, "validation_used": False, "audit_used": False}


def run(root):
    model_remote_only()
    config = policy()
    preserve(root)
    prepare(root / "data")
    data_report, splits = read_splits(root / "data")
    manifest = json.loads((HERE / "data/manifest.json").read_text())
    rows = selected_rows(manifest, splits["train"])
    plans = schedules(rows)
    kit = prepare_kit(root)
    trainer = load_trainer(kit)
    foundation, base, _ = prepare_reference(root)
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    if torch.cuda.is_available():
        raise ValueError("matched controls require the bounded CPU runner")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    node = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    node.no_padding()
    node.no_truncation()
    hp = trainer.Hyperparams(max_length=256, batch_size=2, grad_accum=1,
                            lora_rank=config["lora_rank"], lora_alpha=config["lora_alpha"],
                            lora_dropout=config["lora_dropout"], lr=config["learning_rate"],
                            seed=config["seed"], epochs=3, eval_batch=1, early_stopping=False)
    reports, common_initial = {}, None
    for name, plan in plans.items():
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        model, tokenizer = trainer.build_model(str(base), hp)
        for row in rows:
            for side in ("chosen", "rejected"):
                text = trainer.render(row["prompt"], row[side])
                ids = node.encode(text).ids
                if len(ids) > 256 or tokenizer(text, truncation=False)["input_ids"] != ids:
                    raise ValueError("complete original task/tokenizer mismatch")
        collate = trainer.PairCollator(tokenizer, 256)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=hp.lr, weight_decay=hp.weight_decay)

        def state_hash(model_obj):
            h = hashlib.sha256()
            for key, parameter in model_obj.named_parameters():
                if parameter.requires_grad:
                    h.update(key.encode())
                    h.update(parameter.detach().cpu().numpy().tobytes())
            return h.hexdigest()

        def evaluate(model_obj, tokenizer_obj):
            scores = []
            with isolated_evaluation(model_obj, torch, np), torch.no_grad():
                for row in rows:
                    for side in ("chosen", "rejected"):
                        encoded = tokenizer_obj(trainer.render(row["prompt"], row[side]), truncation=False, return_tensors="pt")
                        score = trainer._reward(model_obj, encoded["input_ids"], encoded["attention_mask"]).item()
                        if not math.isfinite(score):
                            raise ValueError("nonfinite training-control score")
                        scores.append(score)
            return {"metrics": pair_metrics(rows, scores), "scores": scores}

        initial = state_hash(model)
        if common_initial is not None and initial != common_initial:
            raise ValueError("matched controls require identical initialization")
        common_initial = initial
        history = [{"optimizer_step": 0, **evaluate(model, tokenizer)}]
        losses, gradients = [], []
        started = time.monotonic()
        for step, indices in enumerate(plan, 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch = collate([rows[i] for i in indices])
            chosen = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rejected = trainer._reward(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
            loss = trainer.bt_loss(chosen, rejected, batch["margin"])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite matched training loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            losses.append(float(loss.detach()))
            gradients.append(float(gradient))
            optimizer.step()
            if step % config["report_interval"] == 0:
                history.append({"optimizer_step": step, **evaluate(model, tokenizer)})
                reports[name] = {"history": history, "losses": losses, "gradient_norms": gradients,
                                 "optimizer_steps": step, "initial_trainable_sha256": initial,
                                 "current_trainable_sha256": state_hash(model), "trainable_parameters": sum(p.numel() for p in params),
                                 "elapsed_seconds": time.monotonic() - started,
                                 "batch_plan_sha256": digest(json.dumps(plan, separators=(",", ":")).encode())}
                write_json(root / "reciprocal-diagnostic.json", {"status": "TRAINING_DIAGNOSTIC_NOT_A_CANDIDATE",
                           "policy": config, "reports": reports, "train_sha256": data_report["files"]["train"]["sha256"],
                           "source_manifest_sha256": data_report["source_manifest_sha256"], "training_groups": 4,
                           "training_pairs": 8, "validation_scored": False, "audit_scored": False,
                           "commit": os.environ["GITHUB_SHA"], "run_id": int(os.environ["GITHUB_RUN_ID"]),
                           "owner_attempt": os.environ["OWNER_ATTEMPT"], "licence_metadata": licence_metadata(),
                           "registration_allowed": False})
        del optimizer, params, model, tokenizer, batch, chosen, rejected, loss
        gc.collect()
    preserve(root)


if __name__ == "__main__":
    model_remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if not args.root.resolve().is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("training controls must run inside the public runner")
    def timeout(signum, frame):
        raise TimeoutError("bounded reciprocal control timed out")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(40 * 60)
    try:
        run(args.root)
    finally:
        preserve(args.root)
