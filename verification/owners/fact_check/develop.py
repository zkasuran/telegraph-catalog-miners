# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Bounded remote training improvement. Never an audit or registration report."""
import argparse
import hashlib
import json
import math
import os
import random
import signal
import time
from dataclasses import asdict
from pathlib import Path

from data import HERE, TOKENIZER_SHA, digest, render, write_json
from development_policy import (
    EXPOSED_AUDIT_SHA256,
    RECIPE,
    prepare_development,
    quality,
    select,
)
from frozen_baseline import load_frozen_controls
from gates import load_trainer
from licensing import licence_metadata, preserve_artifact
from preserve import preserve
from run import prepare_kit, prepare_reference, remote_only
from runtime_check import compile_tools, preflight
from training_policy import isolated_evaluation


def fit(root):
    remote_only()
    started = time.monotonic()
    preserve(root)
    rows, pins = prepare_development(root / "data")
    kit = prepare_kit(root)
    trainer = load_trainer(kit)
    hp = trainer.Hyperparams(**RECIPE, grad_checkpointing=False, early_stopping=False, gate_min_acc=.8)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(root / "data/sources/tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    for values in rows.values():
        for row in values:
            ids = [tokenizer.encode(render(row["prompt"], row[side])).ids for side in ("chosen", "rejected")]
            if max(map(len, ids)) > 256 or ids[0] == ids[1]:
                raise ValueError("development input is cropped or indistinguishable")
    vectors = json.loads((kit / "float_runtime/testdata/tokenizer_vectors.json").read_text())["vectors"]
    if len(vectors) != 88:
        raise ValueError("complete primary tokenizer vector suite required")
    for vector in vectors:
        if tokenizer.encode(vector["text"], add_special_tokens=vector["add_special_tokens"]).ids != vector["ids"]:
            raise ValueError("primary tokenizer vector mismatch")
    gate = trainer.gate_dataset(rows["train"], hp)
    if not gate["ok"] or gate["metrics"].get("surface_probe_acc") is None:
        raise ValueError("actual reference development dataset gate failed")
    controls, _ = load_frozen_controls(root / "data", rows["train"], tokenizer)
    shortcuts = controls(rows["validation"])
    if max(shortcuts.values()) > .65:
        raise ValueError("development shortcut exceeds the original quality ceiling")
    compile_tools(root, kit)
    preflight(root)
    foundation, base, _ = prepare_reference(root)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    if torch.cuda.is_available():
        raise ValueError("only the assigned free CPU runtime is permitted")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    model, tok = trainer.build_model(str(base), hp)
    for values in rows.values():
        for row in values:
            for side in ("chosen", "rejected"):
                text = render(row["prompt"], row[side])
                if tok(text, truncation=False)["input_ids"] != tokenizer.encode(text).ids:
                    raise ValueError("training and reference tokenizer disagree")
    if digest((foundation / "tokenizer.json").read_bytes()) != TOKENIZER_SHA:
        raise ValueError("foundation tokenizer identity mismatch")

    def state_hash():
        result = hashlib.sha256()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                result.update(name.encode())
                result.update(parameter.detach().cpu().numpy().tobytes())
        return result.hexdigest()

    def save(destination):
        destination.mkdir(exist_ok=True)
        model.save_pretrained(destination)
        trainer.save_reward_head(model, destination)
        head = destination / "reward_head.pt"
        before = torch.load(head, map_location="cpu", weights_only=True)
        for path in sorted(destination.glob("*.safetensors")) + [head]:
            preserve_artifact(path)
        after = torch.load(head, map_location="cpu", weights_only=True)
        if set(before) != set(after) or any(not torch.equal(before[key], after[key]) for key in before):
            raise ValueError("checkpoint licensing changed reward head bytes")
        preserve(root)

    def evaluate():
        with isolated_evaluation(model, torch, np):
            result = trainer.evaluate(model, tok, rows["validation"], hp)
            result["accuracy_by_label"] = {label: trainer.evaluate(model, tok,
                [row for row in rows["validation"] if row["label"] == label], hp)["pairwise_accuracy"]
                for label in ("True", "False")}
        return result

    initial_hash = state_hash()
    untrained = evaluate()
    save(root / "initial")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=hp.lr, weight_decay=hp.weight_decay)
    sampler = torch.Generator(device="cpu").manual_seed(hp.seed)
    loader = DataLoader(rows["train"], batch_size=1, shuffle=True, num_workers=0, generator=sampler,
                        collate_fn=trainer.PairCollator(tok, hp.max_length))
    if len(loader) % hp.grad_accum:
        raise ValueError("incomplete development accumulation group")
    history, best, best_state, best_hash = [], None, None, None
    steps = 0
    for epoch in range(1, hp.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        epoch_started = time.monotonic()
        for index, batch in enumerate(loader):
            chosen = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rejected = trainer._reward(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
            loss = trainer.bt_loss(chosen, rejected, batch["margin"])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite development training loss")
            (loss / hp.grad_accum).backward()
            losses.append(loss.item())
            if (index + 1) % hp.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
        sampler_state = sampler.get_state().clone()
        metrics = evaluate()
        if not torch.equal(sampler_state, sampler.get_state()):
            raise ValueError("development evaluation advanced training sampler")
        best, improved = select(best, epoch, metrics)
        if improved:
            best_state = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
                          if parameter.requires_grad}
            best_hash = state_hash()
            save(root / "adapter")
        save(root / "checkpoint")
        elapsed = time.monotonic() - epoch_started
        history.append({"epoch": epoch, "metrics": metrics, "train_loss_mean": math.fsum(losses) / len(losses),
                        "epoch_seconds": elapsed, "selected": improved, "rng_restored": True})
        write_json(root / "development-selection.json", {
            "status": "DEVELOPMENT_ONLY", "family": "FACT_CHECK", "run_id": int(os.environ["GITHUB_RUN_ID"]),
            "commit": os.environ["GITHUB_SHA"], "history": history, "best": best, "best_trainable_sha256": best_hash,
            "hyperparameters": asdict(hp), "initial_trainable_sha256": initial_hash, "optimizer_steps": steps,
            "split_pins": pins, "source_manifest_sha256": digest((HERE / "sources.json").read_bytes()),
            "tokenizer_sha256": TOKENIZER_SHA, "audit_read": False, "audit_scored": False,
            "excluded_audit_sha256": EXPOSED_AUDIT_SHA256, "untrained": untrained, "shortcuts": shortcuts,
        })
        print(json.dumps({"epoch": epoch, "validation": metrics, "train_loss": history[-1]["train_loss_mean"]}), flush=True)
        if time.monotonic() - started + (hp.epochs - epoch) * elapsed > 270 * 60:
            raise TimeoutError("measured development work exceeds its bounded CPU reserve")
    if len(history) != RECIPE["epochs"] or best_state is None:
        raise ValueError("predeclared development recipe did not complete")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(best_state[name])
    if state_hash() != best_hash:
        raise ValueError("selected development state was not restored exactly")
    save(root / "adapter")
    tok.save_pretrained(root / "adapter")
    (root / "adapter/tokenizer.json").write_bytes((foundation / "tokenizer.json").read_bytes())
    validation = history[best["epoch"] - 1]["metrics"]
    passed = quality(validation, untrained, shortcuts)
    report = json.loads((root / "development-selection.json").read_text())
    report.update({"status": "DEVELOPMENT_PASS_REQUIRES_FRESH_AUDIT" if passed["passed"] else "DEVELOPMENT_FAIL",
                   "quality": passed, "selected_validation": validation, "epochs_completed": len(history),
                   "elapsed_seconds": time.monotonic() - started, "licence_metadata": licence_metadata(),
                   "registration_ready": False, "package_exported": False,
                   "limits": ["Validation is development-exposed, not independent audit.",
                              "Only source-consistency True/False prose was developed.",
                              "No full-package runtime or node qualification is claimed."]})
    write_json(root / "result.json", report)
    preserve(root)
    return 0 if passed["passed"] else 1


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if not args.root.resolve().is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("development outputs must stay in runner temporary storage")
    def stop(signum, frame):
        raise TimeoutError("bounded development job stopped")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(270 * 60)
    try:
        return fit(args.root)
    except Exception as error:
        write_json(args.root / "development-failure.json", {"error": type(error).__name__ + ": " + str(error),
                   "audit_read": False, "registration_ready": False})
        raise
    finally:
        preserve(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
