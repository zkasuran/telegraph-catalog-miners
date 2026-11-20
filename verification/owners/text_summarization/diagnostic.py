# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only training controls. Never audits or emits a registration candidate."""
import argparse
import gc
import hashlib
import json
import os
import random
import signal
from pathlib import Path

from data import HERE, prepare, read_splits, write_json
from gates import load_trainer
from licensing import licence_metadata
from metrics import pair_metrics
from preserve import preserve
from reference import prepare_kit, prepare_reference
from runtime_check import model_remote_only
from training_policy import isolated_evaluation


def training_controls(manifest, train):
    groups = []
    for row in train:
        if row["group_id"] not in groups:
            groups.append(row["group_id"])
        if len(groups) == 4:
            break
    selected = [dict(row) for row in train if row["group_id"] in groups]
    documents = {d["doi"]: d for g in manifest["splits"]["train"] for d in g["documents"]}
    if len(selected) != 8 or any(r["source_id"] not in documents for r in selected):
        raise ValueError("four complete public training groups required")
    single = [dict(row, prompt="Give a concise title for this requested study's complete abstract:\n\n" + documents[row["source_id"]]["abstract"]) for row in selected]
    return {"original_two_reference": selected, "single_requested_reference_control": single}


def run(root):
    model_remote_only()
    preserve(root)
    prepare(root / "data")
    data_report, splits = read_splits(root / "data")
    manifest = json.loads((HERE / "data/manifest.json").read_text())
    controls = training_controls(manifest, splits["train"])
    kit = prepare_kit(root)
    trainer = load_trainer(kit)
    _, base, _ = prepare_reference(root)
    import numpy as np
    import torch
    if torch.cuda.is_available():
        raise ValueError("training diagnostics require the bounded CPU runner")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    hp = trainer.Hyperparams(max_length=256, batch_size=1, grad_accum=1, lora_rank=8, lora_alpha=32,
                            lora_dropout=0.0, lr=.0005, epochs=3, eval_batch=1, early_stopping=False)
    reports = {}
    for name, rows in controls.items():
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        model, tokenizer = trainer.build_model(str(base), hp)
        model.cpu()
        collate = trainer.PairCollator(tokenizer, 256)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=hp.lr, weight_decay=.01)

        def state_hash(model_obj):
            sha = hashlib.sha256()
            for key, parameter in model_obj.named_parameters():
                if parameter.requires_grad:
                    sha.update(key.encode())
                    sha.update(parameter.detach().cpu().numpy().tobytes())
            return sha.hexdigest()

        def evaluate(model_obj, tokenizer_obj):
            scores = []
            with isolated_evaluation(model_obj, torch, np), torch.no_grad():
                for row in rows:
                    for side in ("chosen", "rejected"):
                        encoded = tokenizer_obj(trainer.render(row["prompt"], row[side]), truncation=False, return_tensors="pt")
                        if encoded["input_ids"].shape[1] > 256:
                            raise ValueError("diagnostic source was cropped")
                        scores.append(trainer._reward(model_obj, encoded["input_ids"], encoded["attention_mask"]).item())
            return pair_metrics(rows, scores)

        initial = state_hash(model)
        history = [{"optimizer_step": 0, "metrics": evaluate(model, tokenizer)}]
        losses, gradients = [], []
        randomizer = random.Random(42)
        order = list(range(len(rows)))
        for step in range(1, 129):
            if (step - 1) % len(rows) == 0:
                randomizer.shuffle(order)
            row = rows[order[(step - 1) % len(rows)]]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch = collate([row])
            chosen = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rejected = trainer._reward(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
            loss = trainer.bt_loss(chosen, rejected, batch["margin"])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite diagnostic loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            gradients.append(float(norm))
            losses.append(float(loss.detach()))
            optimizer.step()
            if step % 16 == 0:
                history.append({"optimizer_step": step, "metrics": evaluate(model, tokenizer)})
                reports[name] = {"history": history, "losses": losses, "gradient_norms": gradients,
                                 "initial_trainable_sha256": initial, "current_trainable_sha256": state_hash(model),
                                 "trainable_parameters": sum(p.numel() for p in params)}
                write_json(root / "training-diagnostic.json", {"status": "TRAINING_DIAGNOSTIC_NOT_A_CANDIDATE",
                           "reports": reports, "train_sha256": data_report["files"]["train"]["sha256"],
                           "source_manifest_sha256": data_report["source_manifest_sha256"], "training_groups": 4,
                           "training_pairs": 8, "validation_scored": False, "audit_scored": False,
                           "commit": os.environ["GITHUB_SHA"], "run_id": int(os.environ["GITHUB_RUN_ID"]),
                           "licence_metadata": licence_metadata(), "registration_allowed": False})
        del optimizer, params, model, tokenizer, batch, chosen, rejected, loss
        gc.collect()
    preserve(root)


if __name__ == "__main__":
    model_remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if not args.root.resolve().is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("diagnostic output must be inside the ephemeral public runner")
    def timeout(signum, frame):
        raise TimeoutError("bounded training control timed out")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(40 * 60)
    try:
        run(args.root)
    finally:
        preserve(args.root)
