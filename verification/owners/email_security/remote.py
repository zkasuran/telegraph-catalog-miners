# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""One guarded, bounded public CPU diagnostic. Never register or deploy."""
import argparse
import gc
import importlib.metadata
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

from data import HERE, audit_labels, digest, evaluate_controls, prepare, read_splits, write_json
from scorer import render
from reference import prepare_kit, prepare_reference, source_binding

PREPARATION_SHA = "5b2ba64a11ff141ac96a20c491fc282f917e722183b2ab67c17e6a12c8ff519b"
TRAINER_SHA = "5da2919c66c2d0840104628ab88e98a9c8e5e578686f71ca9be4dcd91bd18d71"


def remote_only():
    wanted = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "zkasuran/telegraph-catalog-miners",
              "GITHUB_EVENT_NAME": "workflow_dispatch", "RUNNER_ENVIRONMENT": "github-hosted",
              "GITHUB_WORKFLOW": "evaluator-owner-email_security"}
    if any(os.environ.get(k) != v for k, v in wanted.items()) or not os.environ.get("RUNNER_TEMP"):
        raise RuntimeError("models run only in the exact controller-owned public CPU workflow")


def validate_recipe(recipe):
    expected = {"family": "EMAIL_SECURITY", "child": "EMAIL_SECURITY_GENERAL", "class": "judgment",
                "rank": 8, "alpha": 32, "max_tokens": 256, "epochs": 2,
                "training_timeout_seconds": 7200, "runtime_timeout_seconds": 1800,
                "workflow_timeout_minutes": 150, "runner": "ubuntu-24.04",
                "repository": "zkasuran/telegraph-catalog-miners", "upstream_dataset_gate_required": True,
                "production_transport_verified": False, "canonical_claim": False}
    if any(recipe.get(key) != value for key, value in expected.items()):
        raise ValueError("fixed recipe changed")
    for key in ("batch_size", "grad_accum", "seed"):
        if type(recipe.get(key)) is not int or recipe[key] != {"batch_size": 2, "grad_accum": 4, "seed": 42}[key]:
            raise ValueError("fixed CPU hyperparameter changed")


def preserve(root):
    from licensing import preserve_artifact
    for path in sorted(root.rglob("*.safetensors")) + sorted(root.rglob("reward_head.pt")):
        # Only checkpoints authored by this experiment, never downloaded base,
        # foundation or the source kit's public artificial parser fixtures.
        if path.relative_to(root).parts[0] not in ("initial", "adapter", "checkpoint", "EMAIL_SECURITY.safetensors"):
            continue
        preserve_artifact(path)


def state_hash(model):
    h = __import__("hashlib").sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            h.update(name.encode())
            h.update(parameter.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def experiment(root):
    remote_only()
    from runtime import get
    repository = json.loads(get("https://api.github.com/repos/zkasuran/telegraph-catalog-miners"))
    if repository.get("private") is not False or repository.get("full_name") != "zkasuran/telegraph-catalog-miners":
        raise RuntimeError("fixed repository must be publicly accessible before model operations")
    write_json(root / "github-repository.json", {"full_name": repository["full_name"], "private": repository["private"],
               "source": "https://api.github.com/repos/zkasuran/telegraph-catalog-miners"})
    recipe = json.loads((HERE / "recipe.json").read_bytes())
    validate_recipe(recipe)
    bound_source = source_binding()
    prepare(root / "data")
    report, splits, frozen = read_splits(root / "data")
    contract = json.loads((HERE / "contract.json").read_bytes())
    if report["files"] != contract["files"] or report["files"] != bound_source["data"]:
        raise ValueError("original immutable grouped fixture bytes changed")
    identity = {"family": "EMAIL_SECURITY", "owner_attempt": contract["run_id"],
                "github_run_id": os.environ["GITHUB_RUN_ID"], "github_run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
                "commit": os.environ["GITHUB_SHA"], "repository": os.environ["GITHUB_REPOSITORY"],
                "recipe_sha256": digest((HERE / "recipe.json").read_bytes()), "data": report["files"],
                "source_scope": report["scope"], "canonical_claim": False}
    from licensing import licence_metadata
    write_json(root / "licence-boundary.json", licence_metadata())
    write_json(root / "result.json", {**identity, "phase": "source_gate", "status": "INCOMPLETE"})
    kit = prepare_kit(root)
    source = kit / "float_runtime/train_adapter.py"
    if digest(source.read_bytes()) != TRAINER_SHA:
        raise ValueError("reference trainer source changed")
    sys.path.insert(0, str(kit))
    from float_runtime import train_adapter as trainer
    hp = trainer.Hyperparams(max_length=256, lora_rank=8, lora_alpha=32, epochs=2,
                            batch_size=2, grad_accum=4, eval_batch=2, early_stopping=False,
                            grad_checkpointing=False, gate_min_acc=.8)
    gate = trainer.gate_dataset(splits["train"], hp)
    write_json(root / "dataset-gate.json", gate)
    if not gate["ok"] or gate["metrics"].get("surface_probe_acc") is None:
        raise ValueError("unmodified upstream dataset gate failed")
    # Primary schema/class verification precedes foundation/model loading.
    from runtime import compile_tools, capture_class
    compile_tools(root, kit)
    captured = capture_class(root)
    foundation, base, kit = prepare_reference(root)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoTokenizer
    from tokenizers import Tokenizer
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import metadata, tensor_names, target_dims, write_package
    if torch.cuda.is_available():
        raise RuntimeError("GPU is outside fixed CPU recipe")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    trainer.check_tokenizer(tok)
    node = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    node.no_padding()
    node.no_truncation()
    token_count, max_tokens = 0, 0
    for rows in splits.values():
        for row in rows:
            ids = []
            for side in ("chosen", "rejected"):
                text = render(row["prompt"], row[side])
                if trainer.render(row["prompt"], row[side]) != text:
                    raise ValueError("literal renderer mismatch")
                encoded = node.encode(text).ids
                if encoded != tok(text, truncation=False)["input_ids"] or len(encoded) > 256:
                    raise ValueError("tokenizer mismatch or decisive truncation")
                ids.append(encoded)
                token_count += 1
                max_tokens = max(max_tokens, len(encoded))
            if ids[0] == ids[1]:
                raise ValueError("retained answers indistinguishable")
    write_json(root / "tokenizer-check.json", {"texts": token_count, "max_observed_tokens": max_tokens,
               "canonical_render": True, "complete_inputs_fit": True})
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    model, _ = trainer.build_model(str(base), hp)
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    initial_hash = state_hash(model)

    def save(path):
        path.mkdir(exist_ok=True)
        model.save_pretrained(path)
        trainer.save_reward_head(model, path)
        preserve(root)

    def evaluate(rows):
        state = random.getstate(), np.random.get_state(), model.training
        with torch.random.fork_rng(devices=[]):
            try:
                model.eval()
                values = []
                with torch.no_grad():
                    for row in rows:
                        for side in ("chosen", "rejected"):
                            batch = tok(render(row["prompt"], row[side]), truncation=False, return_tensors="pt")
                            value = trainer._reward(model, batch["input_ids"], batch["attention_mask"]).item()
                            if not math.isfinite(value):
                                raise ValueError("nonfinite native score")
                            values.append(value)
                wins = [values[2*i] > values[2*i+1] for i in range(len(rows))]
                deltas = torch.tensor([values[2*i] - values[2*i+1] for i in range(len(rows))])
                loss = torch.nn.functional.softplus(-deltas).mean().item()
                metrics = {"pairwise_accuracy": sum(wins) / len(rows), "val_loss": loss,
                           "by_control": {k: sum(w for r, w in zip(rows, wins) if r["control"] == k) /
                                          sum(r["control"] == k for r in rows) for k in ("name", "type")}}
                labels = [audit_labels(row) for row in rows]
                metrics["per_label"] = {k: sum(win for ls, win in zip(labels, wins) if k in ls) /
                                           sum(k in ls for ls in labels) for k in sorted({k for ls in labels for k in ls})}
                return metrics, values
            finally:
                random.setstate(state[0])
                np.random.set_state(state[1])
                model.train(state[2])

    save(root / "initial")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=hp.lr, weight_decay=hp.weight_decay)
    sampler = torch.Generator().manual_seed(hp.seed)
    loader = DataLoader(splits["train"], batch_size=2, shuffle=True, num_workers=0, generator=sampler,
                        collate_fn=trainer.PairCollator(tok, 256))
    if len(loader) % hp.grad_accum:
        raise ValueError("partial gradient accumulation")
    best, best_state, history = None, None, []
    started, steps = time.monotonic(), 0
    for epoch in range(1, 3):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(loader):
            rc = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rr = trainer._reward(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
            loss = trainer.bt_loss(rc, rr, batch["margin"])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            (loss / hp.grad_accum).backward()
            if (index + 1) % hp.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                save(root / "checkpoint")
                write_json(root / "checkpoint.json", {**identity, "phase": "training", "epoch": epoch,
                           "optimizer_steps": steps, "audit_scored": False, "status": "INCOMPLETE"})
                if time.monotonic() - started > recipe["training_timeout_seconds"]:
                    raise TimeoutError("fixed training ceiling")
        validation, _ = evaluate(splits["validation"])
        improved = best is None or validation["val_loss"] < best["val_loss"]
        if improved:
            best = {"epoch": epoch, "val_loss": validation["val_loss"]}
            best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        history.append({"epoch": epoch, "metrics": validation, "selected": improved})
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(best_state[name])
    save(root / "adapter")
    candidate_hash = state_hash(model)
    write_json(root / "candidate-freeze.json", {**identity, "candidate_state_sha256": candidate_hash,
               "selected": best, "validation_history": history, "audit_scored": False, "fit_split": "train"})
    trained, native = evaluate(splits["audit"])
    write_json(root / "native-audit.json", {"metrics": trained, "scores": native,
               "candidate_state_sha256": candidate_hash, "audit_sha256": report["files"]["audit"]["sha256"]})
    cfg = AutoConfig.from_pretrained(base, local_files_only=True)
    tensors, params = read_peft(root / "adapter", cfg)
    from runtime import HASHES
    meta = metadata(cls="judgment", intent="EMAIL_SECURITY", foundation_hash=HASHES["foundation_lora_fp32.onnx"],
                    foundation_version="modernbert-base-lora16-v1", rank=params["rank"], alpha=params["alpha"],
                    target_modules=params["target_modules"], max_tokens=256,
                    training={"family": "EMAIL_SECURITY", "children": [{"intent": "EMAIL_SECURITY_GENERAL",
                              "dataset_version": report["files"]["audit"]["sha256"], "pairs": 128,
                              "accuracy": trained["pairwise_accuracy"], "held_out": True}], "source_scope": report["scope"],
                              "preference_authorship": "AI-assisted constructed protocol fixtures. No human review claimed."})
    meta.update(licence_metadata())
    package = root / "EMAIL_SECURITY.safetensors"
    write_package(package, meta, {n: tensors[n] for n in tensor_names(cfg.num_hidden_layers)})
    write_json(root / "foundation.json", {"foundation_hash": meta["foundation_hash"],
               "foundation_version": meta["foundation_version"], "tokenizer_hash": meta["tokenizer_hash"],
               "base_model": str(base), "r_max": 16, "num_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
               "targets": {k: list(v) for k, v in target_dims(cfg).items()}})
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(initial[name])
    if state_hash(model) != initial_hash:
        raise ValueError("untrained comparator state changed")
    untrained, _ = evaluate(splits["audit"])
    controls = evaluate_controls(splits["audit"], frozen)
    length_diffs = []
    for row in splits["audit"]:
        ids = [node.encode(render(row["prompt"], row[side])).ids for side in ("chosen", "rejected")]
        length_diffs.append(len(ids[0]) - len(ids[1]))
    controls["token_length"] = sum(1 if d > 0 else .5 if d == 0 else 0 for d in length_diffs) / len(length_diffs)
    # Measure evidence ablation without touching the candidate selection.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(best_state[name])
    if state_hash(model) != candidate_hash:
        raise ValueError("frozen candidate changed while measuring untrained baseline")
    ablated = [{**r, "prompt": "EMAIL_SECURITY_GENERAL request."} for r in splits["audit"]]
    answer_only, _ = evaluate(ablated)
    model = None
    del initial, best_state
    gc.collect()
    from runtime import validate_package, execute_runtime
    manifest = validate_package(root)
    runtime = execute_runtime(root, manifest, splits["audit"], native)
    failures = []
    if trained["pairwise_accuracy"] < .8 or min(trained["per_label"].values()) < .7:
        failures.append("audit ordering or query/type subset gate")
    if trained["pairwise_accuracy"] < untrained["pairwise_accuracy"] + .05:
        failures.append("less than five points above untrained")
    if controls["answer_only"] > .65 or controls["length"] > .65 or answer_only["pairwise_accuracy"] > .65:
        failures.append("answer-only or length shortcut")
    if trained["pairwise_accuracy"] + 1e-9 < max(controls["literal_word_overlap"], controls["token_length"]) + .10:
        failures.append("less than ten points above strongest overlap/token-length shortcut")
    if not runtime["passed"]:
        failures.append("exported runtime gate")
    result = {**identity, "status": "PASS" if not failures else "FAIL", "failures": failures,
              "hyperparameters": asdict(hp), "trained": trained, "untrained": untrained, "controls": controls,
              "answer_only_ablation": answer_only, "class_snapshot": captured,
              "candidate_state_sha256": candidate_hash, "package_sha256": digest(package.read_bytes()),
              "runtime": runtime, "node_qualification": "not attempted",
              "versions": {key: importlib.metadata.version(key) for key in ("torch", "transformers", "peft", "numpy", "tokenizers", "safetensors")}}
    write_json(root / "result.json", result)
    from ready import build_ready
    hashes = {"source": digest((HERE / "source-manifest.json").read_bytes()),
              "candidate": digest((root / "candidate-freeze.json").read_bytes()),
              "runtime": digest((root / "runtime-result.json").read_bytes()),
              "native_audit": digest((root / "native-audit.json").read_bytes())}
    if not failures:
        ready = build_ready(identity, report, manifest, runtime, untrained, controls, answer_only, hashes)
        write_json(root / "registration-ready.json", ready)
        print(json.dumps({"status": ready["status"], "run_id": ready["run_id"], "package_sha256": ready["package_sha256"]}))
    return 1 if failures else 0


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    root = parser.parse_args().root
    if root.exists():
        raise ValueError("fresh remote run directory required; preserve previous results")
    root.mkdir(parents=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError("workflow timeout")))
    try:
        return experiment(root)
    except Exception as error:
        write_json(root / "failure.json", {"status": "FAIL", "error": str(error), "node_qualification": "not attempted"})
        raise
    finally:
        preserve(root)


if __name__ == "__main__":
    raise SystemExit(main())
