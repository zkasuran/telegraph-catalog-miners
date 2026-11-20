# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Fixed remote CPU experiment. No registration or local model execution."""
import argparse
import gc
import json
import math
import os
import random
import re
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

from audit_policy import development_gate
from curriculum import check_tokens
from data import HERE, VIEWS, prepare, read_splits, view, write_json
from gates import run_gate
from licensing import OWN_ID, licence_metadata, preserve_artifact
from metrics import all_formats, pair_metrics
from preserve import preserve
from primary import verify_primary
from reference import prepare_kit, prepare_reference
from report import registration_ready
from role_bootstrap import (
    WARMUP_EPOCHS,
    WARMUP_LR,
    WARMUP_STEPS,
    admit,
    warmup_plan,
    warmup_rows,
)
from runtime_check import (
    compile_tools,
    execute_runtime,
    model_remote_only,
    preflight,
    validate_package,
)
from static_check import main as check_static
from training_policy import EPOCHS, development_selection, isolated_evaluation


def remote_only():
    model_remote_only()
    if (os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_REPOSITORY") != "zkasuran/telegraph-catalog-miners"
            or not os.environ.get("RUNNER_TEMP") or not re.fullmatch(r"[a-f0-9]{32}", os.environ.get("OWNER_ATTEMPT", ""))):
        raise RuntimeError("models run only on the assigned public GitHub CPU workflow")


def interval(wins, n):
    z = 1.959963984540054
    p = wins / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [center - half, center + half]


def experiment(root):
    remote_only()
    if (HERE / "frozen-candidate.json").exists():
        raise ValueError("candidate is frozen after its audit; only byte-preserving recovery is allowed")
    started = time.monotonic()
    preserve(root)
    check_static()
    data_root = root / "data"
    prepare(data_root)
    primary = verify_primary(data_root / "xml")
    kit = prepare_kit(root)
    trainer, hp, splits, controls, gate = run_gate(data_root, kit)
    data_report, _ = read_splits(data_root)
    write_json(root / "result.json", {"status": "INCOMPLETE", "phase": "data_gate", "dataset_gate": gate,
               "data": data_report, "runtime_integration": "INCOMPLETE"})
    if not gate["ok"] or not primary["grants_verified"]:
        return 1
    data_report["training_allowed"] = True
    data_report["authorization"] = "Source grants and the unchanged full dataset gate passed before model loading."
    # Bind the live primary family/child class before any model loading.
    compile_tools(root, kit)
    preflight(root)
    foundation, base, kit = prepare_reference(root)
    import numpy as np
    import torch
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import (
        metadata,
        target_dims,
        tensor_names,
        write_package,
    )
    from runtime_check import RELEASE_HASHES as FOUNDATION
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoTokenizer
    if torch.cuda.is_available():
        raise RuntimeError("fixed CPU experiment refuses GPU runtime")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if not trainer.TOKENIZER_VECTORS.is_file():
        raise ValueError("missing primary tokenizer vectors")
    trainer.check_tokenizer(tok)
    from tokenizers import Tokenizer
    node = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    node.no_padding()
    node.no_truncation()
    for rows in splits.values():
        for row in rows:
            for side in ("chosen", "rejected"):
                text = trainer.render(row["prompt"], row[side])
                if tok(text, truncation=False)["input_ids"] != node.encode(text).ids:
                    raise ValueError("training and released tokenizer mismatch")

    def seed():
        random.seed(hp.seed)
        np.random.seed(hp.seed)
        torch.manual_seed(hp.seed)

    def state_hash(model):
        h = __import__("hashlib").sha256()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                h.update(name.encode())
                h.update(parameter.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def evaluate(model, rows):
        with isolated_evaluation(model, torch, np):
            metrics = trainer.evaluate(model, tok, rows, hp)
            metrics["subsets"] = {}
            for name, predicate in (("chosen_longer", lambda r: len(r["chosen"]) > len(r["rejected"])),
                                    ("chosen_shorter", lambda r: len(r["chosen"]) < len(r["rejected"]))):
                subset = [r for r in rows if predicate(r)]
                metrics["subsets"][name] = trainer.evaluate(model, tok, subset, hp)["pairwise_accuracy"]
            metrics["pair_wilson_95_not_group_independent"] = interval(round(metrics["pairwise_accuracy"] * len(rows)), len(rows))
        return metrics

    def save_checkpoint(model, destination):
        destination.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(destination)
        trainer.save_reward_head(model, destination)
        head = destination / "reward_head.pt"
        original = torch.load(head, map_location="cpu", weights_only=True)
        for path in sorted(destination.glob("*.safetensors")) + [head]:
            preserve_artifact(path)
        restored = torch.load(head, map_location="cpu", weights_only=True)
        if original.keys() != restored.keys() or any(not torch.equal(original[k], restored[k]) for k in original):
            raise ValueError("checkpoint ZIP licence record changed the reward head")
        preserve(root)

    def native_scores(model, rows):
        scores = []
        model.eval()
        with torch.no_grad():
            for row in rows:
                for side in ("chosen", "rejected"):
                    encoded = tok(trainer.render(row["prompt"], row[side]), truncation=False, return_tensors="pt")
                    value = trainer._reward(model, encoded["input_ids"], encoded["attention_mask"]).item()
                    if not math.isfinite(value):
                        raise ValueError("nonfinite native audit score")
                    scores.append(value)
        return scores

    seed()
    model, tok = trainer.build_model(str(base), hp)
    initial_hash = state_hash(model)
    save_checkpoint(model, root / "initial")
    development_views = {style: [view(r, style) for r in splits["validation"]] for style in VIEWS}
    with isolated_evaluation(model, torch, np):
        untrained_development = {style: pair_metrics(rows, native_scores(model, rows)) for style, rows in development_views.items()}
    # Complete all nine fixed main epochs. Only development loss selects the
    # checkpoint. Its extra evaluation consumes no training RNG.
    if hp.epochs != EPOCHS:
        raise ValueError("fixed nine-epoch training contract changed")
    params = [p for p in model.parameters() if p.requires_grad]
    manifest_source = json.loads((HERE / "data/manifest.json").read_text())
    warm = warmup_rows(manifest_source, splits["train"])
    warm_token_report = check_tokens(warm, node, trainer.render)
    for row in warm:
        for side in ("chosen", "rejected"):
            text = trainer.render(row["prompt"], row[side])
            if tok(text, truncation=False)["input_ids"] != node.encode(text).ids:
                raise ValueError("curriculum tokenizer mismatch")
    warm_optimizer = torch.optim.AdamW(params, lr=WARMUP_LR, weight_decay=hp.weight_decay)
    warm_schedule = warmup_plan(warm)
    warm_collate = trainer.PairCollator(tok, 256)
    warm_losses, warm_steps = [], 0
    warm_started = time.monotonic()
    for epoch_index in range(WARMUP_EPOCHS):
        model.train()
        warm_optimizer.zero_grad(set_to_none=True)
        for indices in warm_schedule[epoch_index * 4:(epoch_index + 1) * 4]:
            batch = warm_collate([warm[i] for i in indices])
            rc = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rr = trainer._reward(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
            warm_loss = trainer.bt_loss(rc, rr, batch["margin"])
            if not torch.isfinite(warm_loss):
                raise ValueError("nonfinite train-only curriculum loss")
            warm_loss.backward()
            warm_losses.append(float(warm_loss.detach()))
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            warm_optimizer.step()
            warm_optimizer.zero_grad(set_to_none=True)
            warm_steps += 1
            if warm_steps % 64 == 0:
                save_checkpoint(model, root / "warmup")
                write_json(root / "bootstrap-progress.json", {"status": "INCOMPLETE", "optimizer_steps": warm_steps,
                           "planned_steps": WARMUP_STEPS, "unique_training_pairs": len(warm), "audit_scored": False,
                           "source_manifest_sha256": data_report["source_manifest_sha256"], "trainable_sha256": state_hash(model)})
    if warm_steps != WARMUP_STEPS:
        raise ValueError("complete fixed warm-up accumulation groups required")
    with isolated_evaluation(model, torch, np):
        bootstrap_metrics = pair_metrics(warm, native_scores(model, warm))
    curriculum_report = {"recipe_version": 4, "training_pairs": len(warm), "unique_training_groups": 4,
               "training_presentations": sum(map(len, warm_schedule)), "training_only": True, "epochs": WARMUP_EPOCHS,
               "view": "original_complete_two_reference", "training_fit": bootstrap_metrics,
               "optimizer_steps": warm_steps, "learning_rate": WARMUP_LR, "mean_loss": sum(warm_losses) / len(warm_losses),
                "training_sha256": data_report["files"]["train"]["sha256"], "audit_scored": False,
                "source_manifest_sha256": data_report["source_manifest_sha256"], "validation_scored": False,
                "elapsed_seconds": time.monotonic() - warm_started, **warm_token_report,
                "initial_trainable_sha256": initial_hash, "post_warmup_trainable_sha256": state_hash(model),
                "commit": os.environ["GITHUB_SHA"], "run_id": os.environ["GITHUB_RUN_ID"],
                "owner_attempt": os.environ["OWNER_ATTEMPT"]}
    # validation_scored describes this warm-up only; the seed-matched development
    # baseline above was measured before any fitted updates.
    write_json(root / "curriculum.json", curriculum_report)
    save_checkpoint(model, root / "warmup")
    if not admit(bootstrap_metrics):
        write_json(root / "result.json", {"status": "ROLE_BOOTSTRAP_GATE_FAIL", "recipe_version": 4,
                   "phase": "main_training_not_started_clean_audit_unscored", "curriculum": curriculum_report,
                   "data": data_report, "audit_scored": False, "commit": os.environ["GITHUB_SHA"],
                   "run_id": os.environ["GITHUB_RUN_ID"], "owner_attempt": os.environ["OWNER_ATTEMPT"]})
        preserve(root)
        return 1
    del warm_optimizer, warm_schedule, warm_collate, warm
    optimizer = torch.optim.AdamW(params, lr=hp.lr, weight_decay=hp.weight_decay)
    sampler = torch.Generator(device="cpu").manual_seed(hp.seed)
    loader = DataLoader(splits["train"], batch_size=1, shuffle=True,
                        collate_fn=trainer.PairCollator(tok, 256), num_workers=0, generator=sampler)
    if len(loader) % hp.grad_accum:
        raise ValueError("fixed CPU batch requires complete accumulation groups")
    timings, losses = [], []
    best, best_state, best_hash = None, None, None
    development_history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_started = time.monotonic()
        for i, batch in enumerate(loader):
            rc = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rr = trainer._reward(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
            loss = trainer.bt_loss(rc, rr, batch["margin"])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            (loss / hp.grad_accum).backward()
            losses.append(loss.item())
            if (i + 1) % hp.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                timings.append(time.monotonic() - step_started)
                save_checkpoint(model, root / "checkpoint")
                write_json(root / "checkpoint.json", {"status": "INCOMPLETE", "optimizer_steps": len(timings),
                           "step_seconds": timings, "hyperparameters": asdict(hp), "initial_sha256": initial_hash,
                           "runtime_integration": "INCOMPLETE", "epoch": epoch, "max_epochs": EPOCHS,
                           "selected_development": best, "licence_metadata": licence_metadata()})
                remaining = EPOCHS * len(loader) / hp.grad_accum - len(timings)
                if time.monotonic() - started + remaining * max(timings) > 300 * 60:
                    raise TimeoutError("measured throughput exceeds fixed training reserve")
                step_started = time.monotonic()
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state().clone()
        sampler_rng = sampler.get_state().clone()
        development = evaluate(model, splits["validation"])
        numpy_after = np.random.get_state()
        if (random.getstate() != python_rng or numpy_rng[0] != numpy_after[0]
                or not np.array_equal(numpy_rng[1], numpy_after[1]) or numpy_rng[2:] != numpy_after[2:]
                or not torch.equal(torch_rng, torch.get_rng_state()) or not torch.equal(sampler_rng, sampler.get_state())):
            raise ValueError("development evaluation advanced training RNG")
        best, improved = development_selection(best, epoch, development)
        if improved:
            best_state = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
            best_hash = state_hash(model)
            save_checkpoint(model, root / "adapter")
        development_history.append({"epoch": epoch, "metrics": development, "selected": improved,
                                    "rng_restored": True, "split": "validation"})
        write_json(root / "development-selection.json", {"history": development_history, "best": best,
                   "best_trainable_sha256": best_hash, "selection_split": "validation", "selection_metric": "val_loss",
                   "audit_scored": False, "epochs_required": EPOCHS, "epochs_completed": epoch})
    if best_state is None or len(development_history) != EPOCHS:
        raise ValueError("nine-epoch development selection did not complete")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(best_state[name])
    if state_hash(model) != best_hash:
        raise ValueError("restored checkpoint differs from the development-selected state")
    validation = development_history[best["epoch"] - 1]["metrics"]
    save_checkpoint(model, root / "adapter")
    tok.save_pretrained(root / "adapter")
    (root / "adapter/tokenizer.json").write_bytes((foundation / "tokenizer.json").read_bytes())
    with isolated_evaluation(model, torch, np):
        trained_development = {style: pair_metrics(rows, native_scores(model, rows)) for style, rows in development_views.items()}
    development_controls = {style: controls(rows) for style, rows in development_views.items()}
    admission = development_gate(trained_development, untrained_development, development_controls)
    write_json(root / "development-gate.json", {"gate": admission, "trained": trained_development,
               "untrained": untrained_development, "controls": development_controls,
               "selected_trainable_sha256": best_hash, "audit_scored": False})
    if not admission["audit_allowed"]:
        write_json(root / "result.json", {"status": "DEVELOPMENT_GATE_FAIL", "family": "TEXT_SUMMARIZATION",
                   "phase": "training_complete_clean_audit_unscored", "development": admission,
                   "data": data_report, "hyperparameters": asdict(hp), "epochs_completed": EPOCHS,
                   "optimizer_steps": len(timings), "train_loss_mean": sum(losses) / len(losses),
                   "selected_epoch": best["epoch"], "selected_trainable_sha256": best_hash,
                   "selection_split": "validation", "audit_scored": False,
                   "commit": os.environ["GITHUB_SHA"], "run_id": os.environ["GITHUB_RUN_ID"],
                    "owner_attempt": os.environ["OWNER_ATTEMPT"], "node_qualification": "not attempted"})
        failed = json.loads((root / "result.json").read_text())
        failed.update({"recipe_version": 4, "curriculum": curriculum_report,
                       "total_optimizer_steps": warm_steps + len(timings)})
        write_json(root / "result.json", failed)
        preserve(root)
        return 1
    audit_views = {style: [view(r, style) for r in splits["audit"]] for style in VIEWS}
    native = {style: native_scores(model, rows) for style, rows in audit_views.items()}
    trained = {style: pair_metrics(audit_views[style], scores) for style, scores in native.items()}
    write_json(root / "native-audit-scores.json", {"scores": native, "audit_sha256": data_report["files"]["audit"]["sha256"],
               "scope": "same complete original rendering, native fp32 reference", "checkpoint_fixed_before_audit": True})
    cfg = AutoConfig.from_pretrained(base, local_files_only=True)
    tensors, params = read_peft(root / "adapter", cfg)
    write_json(root / "foundation.json", {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
               "foundation_version": "modernbert-base-lora16-v1", "tokenizer_hash": FOUNDATION["tokenizer.json"],
               "base_model": str(base), "r_max": 16, "num_layers": cfg.num_hidden_layers,
               "hidden_size": cfg.hidden_size, "targets": {k: list(v) for k, v in target_dims(cfg).items()}})
    meta = metadata(cls="judgment", intent="TEXT_SUMMARIZATION", foundation_hash=FOUNDATION["foundation_lora_fp32.onnx"],
                    foundation_version="modernbert-base-lora16-v1", rank=params["rank"], alpha=params["alpha"],
                    target_modules=params["target_modules"], max_tokens=256,
                    training={"family": "TEXT_SUMMARIZATION", "children": [{"intent": "TEXT_SUMMARIZATION_GENERAL",
                               "dataset_version": data_report["files"]["audit"]["sha256"], "pairs": len(splits["audit"]),
                               "accuracy": trained["raw"]["pairwise_accuracy"], "held_out": True}],
                               "trainer": "Fixed source-conditioned title relevance experiment", "backbone": "ModernBERT-base",
                              "backbone_version": "8949b909ec900327062f0ebf497f51aef5e6f0c8"})
    meta["x-source-provenance"] = json.dumps({"data": data_report, "hyperparameters": asdict(hp),
                                             "recipe_version": 4, "curriculum": curriculum_report}, separators=(",", ":"))
    meta.update(licence_metadata())
    write_package(root / "TEXT_SUMMARIZATION.safetensors", meta, {n: tensors[n] for n in tensor_names(cfg.num_hidden_layers)})
    preserve_artifact(root / "TEXT_SUMMARIZATION.safetensors")
    del model, tensors, optimizer, params, loader, sampler, rc, rr, loss, best_state
    gc.collect()
    seed()
    initial, _ = trainer.build_model(str(base), hp)
    if state_hash(initial) != initial_hash:
        raise ValueError("seed-matched untrained initialization mismatch")
    untrained = {style: pair_metrics(rows, native_scores(initial, rows)) for style, rows in audit_views.items()}
    del initial
    gc.collect()
    frozen_audit = {style: controls(rows) for style, rows in audit_views.items()}
    report = {"status": "INCOMPLETE", "phase": "experiment_completed_runtime_pending", "family": "TEXT_SUMMARIZATION",
              "dataset_gate": gate, "data": data_report, "hyperparameters": asdict(hp), "trained": trained,
              "untrained": untrained, "validation": validation, "audit_controls": frozen_audit,
               "quality": all_formats(trained, untrained, frozen_audit), "initial_sha256": initial_hash,
              "selected_epoch": best["epoch"], "selection_split": "validation", "development_history": development_history,
              "selected_trainable_sha256": best_hash, "epochs_completed": EPOCHS, "evaluation_rng_restored": True,
              "licence_metadata": licence_metadata(), "own_additions_licence": OWN_ID,
              "optimizer_steps": len(timings), "step_seconds": timings, "train_loss_mean": sum(losses) / len(losses),
              "elapsed_seconds": time.monotonic() - started, "runtime_integration": "INCOMPLETE",
              "runtime_gap": "Full live-class parser and reference-runtime checks have not completed yet.",
               "node_qualification": "not attempted", "commit": os.environ["GITHUB_SHA"], "run_id": os.environ["GITHUB_RUN_ID"]}
    report["owner_attempt"] = os.environ["OWNER_ATTEMPT"]
    report.update({"recipe_version": 4, "curriculum": curriculum_report,
                   "total_optimizer_steps": warm_steps + len(timings)})
    write_json(root / "result.json", report)
    manifest = validate_package(root)
    runtime = execute_runtime(root, manifest, native)
    passed = report["quality"]["passed"] and runtime["verdict"] == "PASS"
    report.update({"status": "EXPERIMENT_PASS_NODE_UNTESTED" if passed else "EXPERIMENT_FAIL",
                   "phase": "complete_source_experiment", "verdict": "PASS" if passed else "FAIL",
                   "runtime_integration": "PASS" if runtime["verdict"] == "PASS" else "FAIL",
                   "runtime": runtime, "package_manifest": manifest,
                   "elapsed_seconds": time.monotonic() - started})
    report.pop("runtime_gap", None)
    write_json(root / "result.json", report)
    if passed:
        registration_ready(root, report, primary)
        write_json(root / "candidate-manifest.json", {"status": "SOURCE_CANDIDATE_PENDING_CONTROLLER_NODE_TEST",
                   "SPDX-License-Identifier": OWN_ID, "licence_metadata": licence_metadata(),
                   "family": "TEXT_SUMMARIZATION", "child": "TEXT_SUMMARIZATION_GENERAL", "package_sha256": manifest["sha256"],
                   "keccak256": manifest["keccak256"], "audit_sha256": data_report["files"]["audit"]["sha256"],
                   "commit": os.environ["GITHUB_SHA"], "run_id": os.environ["GITHUB_RUN_ID"],
                   "source_scope": data_report["scope"], "node_qualification": "not attempted", "registration": "not attempted"})
    preserve(root)
    return 0 if passed else 1


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if not args.root.resolve().is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("output must be inside runner temporary storage")
    def stop(signum, frame):
        raise TimeoutError("bounded job stopped by signal " + str(signum))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(300 * 60)
    try:
        return experiment(args.root)
    except Exception as exc:
        preserve(args.root)
        write_json(args.root / "failure.json", {"status": "INCOMPLETE", "error": type(exc).__name__ + ": " + str(exc),
                   "SPDX-License-Identifier": OWN_ID, "licence_metadata": licence_metadata()})
        raise
    finally:
        preserve(args.root)


if __name__ == "__main__":
    sys.exit(main())
