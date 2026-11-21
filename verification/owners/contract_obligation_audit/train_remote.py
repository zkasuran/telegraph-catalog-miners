# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Train/validation freeze without inspecting the independent audit. CPU only."""
import argparse
import gc
import importlib.metadata
import os
import random
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

from baselines import validate_frozen
from contracts import FAMILY, FOUNDATION, HERE, check_config, file_sha, read_jsonl, remote_only, render, strict_json, write_json
from data import pair
from prepare_remote import compile_runtime, full_parser, prepare
from validation_policy import require_validation, warmstart_policy
from parent_checkpoint import PARENT_CHECKPOINT_SHA, PARENT_COMMIT, PARENT_RUN, check_candidate_tensor, restore
from source_cues import measure_source_cues
from rank_growth import grow, verify_trained
from paired_objective import OBJECTIVES, paired_training


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--objective", choices=OBJECTIVES, default=os.environ.get("CONTRACT_TRAIN_OBJECTIVE", "independent"))
    args = parser.parse_args()
    if args.objective not in OBJECTIVES:
        raise ValueError("unknown source training objective")
    root = args.root.resolve()
    config = check_config(strict_json((HERE/"config.json").read_bytes()))
    train, validation = [read_jsonl(root/"data"/(name+".jsonl")) for name in ("train", "validation")]
    source = strict_json((root/"data/source-manifest.json").read_bytes())
    for name in ("train", "validation", "audit"):
        if file_sha(root/"data"/(name+".jsonl")) != source["files"][name]["sha256"]:
            raise ValueError("frozen source data identity changed")
    validate_frozen(strict_json((HERE/"frozen-baselines-v8.json").read_bytes()), train)
    foundation, base, kit = prepare(root, weights=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Neural imports happen only after the remote guard and source checks.
    import numpy as np
    import torch
    from float_runtime import train_adapter as trainer
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import metadata, target_dims, tensor_names, write_package
    from tokenizers import Tokenizer
    from transformers import AutoConfig, AutoTokenizer
    if torch.cuda.is_available():
        raise RuntimeError("this pipeline permits CPU only")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    policy = warmstart_policy()
    if policy["epochs"] != config["epochs"] or policy["max_tokens"] != config["max_tokens"]:
        raise ValueError("source and optimizer policy mismatch")
    hp = trainer.Hyperparams(max_length=config["max_tokens"], batch_size=2, grad_accum=4, epochs=config["epochs"],
                             eval_batch=2, gate_min_acc=.80, grad_checkpointing=False, seed=config["seed"],
                             lr=policy["learning_rate"], weight_decay=policy["weight_decay"],
                             lora_rank=policy["rank"], lora_alpha=policy["alpha"],
                              es_patience=policy["patience"], es_metric="pairwise_accuracy")
    if args.objective == "paired-worst-format":
        hp.batch_size = 1  # One original emits two adjacent views, same two-preference batch.
    gate = trainer.gate_dataset(train, hp)
    write_json(root/"upstream-dataset-gate.json", gate)
    if gate["ok"] is not True:
        raise ValueError("unmodified upstream dataset gate failed before model loading")
    training_views = [pair(row, style) for row in train for style in ("json", "prose")]
    view_gate = trainer.gate_dataset(training_views, hp)
    write_json(root/"upstream-training-view-gate.json", view_gate)
    if view_gate["ok"] is not True:
        raise ValueError("unmodified upstream multi-format view gate failed before model loading")
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    node_tok = Tokenizer.from_file(str(foundation/"tokenizer.json"))
    node_tok.no_padding()
    node_tok.no_truncation()
    if not trainer.TOKENIZER_VECTORS.exists():
        raise ValueError("actual tokenizer conformance vectors are absent")
    trainer.check_tokenizer(tok)
    checked = 0
    for rows in (train, validation):
        for original in rows:
            for style in ("json", "prose"):
                row = pair(original, style)
                for side in ("chosen", "rejected"):
                    text = render(row["prompt"], row[side])
                    if text != trainer.render(row["prompt"], row[side]):
                        raise ValueError("literal public rendering changed")
                    ids = tok(text, truncation=False)["input_ids"]
                    if len(ids) > hp.max_length or ids != node_tok.encode(text).ids:
                        raise ValueError("tokenization mismatch or truncated differing claim")
                    checked += 1
    write_json(root/"tokenizer-check.json", {"actual_rendered_texts_checked": checked,
                "tokenizer_sha256": FOUNDATION["tokenizer.json"], "audit_read_for_selection": False})
    cfg = AutoConfig.from_pretrained(base, local_files_only=True)
    licence = strict_json((HERE/"licences.json").read_bytes())
    def export(adapter, name):
        tensors, params = read_peft(adapter, cfg)
        training = {"family": FAMILY, "children": [{"intent": FAMILY+"_GENERAL", "dataset_version": file_sha(root/"data/source-manifest.json"),
                    "pairs": len(train), "held_out": False}], "source_scope": "Four audited NDA clause claims plus five auxiliary training claims, not performance compliance",
                    "auxiliary_training": {"pairs": source["auxiliary_training_pairs"], "hypotheses": source["auxiliary_training_hypotheses"], "audited_claim": False},
                    "wider_training": {"pairs": source["wider_training_pairs"], "render": "truthful hash citations, complete original prompt spans", "audited_claim": False},
                    "preference_authorship": "Published document-level source labels and AI-assisted response templates, not human preference reviews"}
        if args.objective == "paired-worst-format":
            training["objective"] = "paired worst-format Bradley-Terry, unchanged margin regularization"
        meta = metadata(cls="judgment", intent=FAMILY, foundation_hash=FOUNDATION["foundation_lora_fp32.onnx"],
                        foundation_version="modernbert-base-lora16-v1", rank=params["rank"], alpha=params["alpha"],
                        target_modules=params["target_modules"], max_tokens=hp.max_length, training=training)
        meta.update({"x-upstream-notice": (HERE/"NOTICE").read_text(), "x-upstream-apache-2.0": licence["apache_2_0"],
                     "x-upstream-mit": licence["kit_mit"], "x-upstream-cc-by-4.0": licence["cc_by_4_0"], "x-licence": licence["sand"]})
        package = root/(name+".safetensors")
        write_package(package, meta, {n: tensors[n] for n in tensor_names(cfg.num_hidden_layers)})
        return package
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    initial, _ = trainer.build_model(str(base), hp)
    from baselines import measure
    frozen_baseline = strict_json((HERE/"frozen-baselines-v8.json").read_bytes())
    source_baseline = strict_json((HERE/"frozen-source-cues-v8.json").read_bytes())
    if source_baseline["training_data_sha256"] != source["files"]["train"]["sha256"]:
        raise ValueError("source-aware baseline is not bound to this training data")
    def validation_metrics(model):
        formats = {}
        for style in ("json", "prose"):
            rows = [pair(row, style) for row in validation]
            measured = trainer.evaluate(model, tok, rows, hp)
            formats[style] = {"accuracy": measured["pairwise_accuracy"], "per_label": {
                label: trainer.evaluate(model, tok, [row for row in rows if row["label"] == label], hp)["pairwise_accuracy"]
                for label in ("Entailment", "Contradiction")}, "original_pairs": len(validation), "fit_split": "validation",
                "shortcut_accuracy": max([*measure(rows, frozen_baseline).values(), *measure_source_cues(rows, source_baseline).values()])}
        return formats
    initial_validation = validation_metrics(initial)
    initial_path = root/"untrained-adapter"
    initial.save_pretrained(initial_path)
    trainer.save_reward_head(initial, initial_path)
    export(initial_path, "untrained")
    del initial
    gc.collect()
    trained_path = root/"trained-adapter"
    parent = grow(restore(root), root, cfg, hp.seed)
    original_build, original_evaluate, original_loss = trainer.build_model, trainer.evaluate, trainer.bt_loss
    def resumed_build(base_model, hyperparams):
        model, tokenizer = original_build(base_model, hyperparams)
        model.load_adapter(str(parent), adapter_name="default", is_trainable=True)
        model.set_adapter("default")
        return model, tokenizer
    validation_views = [{**pair(row, style), "view_style": style} for row in validation for style in ("json", "prose")]
    def worst_format_evaluate(model, tokenizer, rows, hyperparams, negatives=None):
        measured = original_evaluate(model, tokenizer, rows, hyperparams, negatives)
        if rows and {row.get("view_style") for row in rows} == {"json", "prose"}:
            views = [original_evaluate(model, tokenizer, [row for row in rows if row["view_style"] == style], hyperparams, negatives)
                     for style in ("json", "prose")]
            measured["val_loss"] = max(view["val_loss"] for view in views)
            measured["pairwise_accuracy"] = min(view["pairwise_accuracy"] for view in views)
        return measured
    # Best epoch selection sees validation only and protects the worse format.
    def regularized_loss(chosen_reward, rejected_reward, margin):
        return original_loss(chosen_reward, rejected_reward, margin) + policy["reward_margin_penalty"]*(chosen_reward-rejected_reward).float().square().mean()
    trainer.build_model, trainer.evaluate, trainer.bt_loss = resumed_build, worst_format_evaluate, regularized_loss
    try:
        context, training_input = nullcontext(), training_views
        if args.objective == "paired-worst-format":
            from paired_objective_remote import coupled_training, gradient_smoke
            gradient_smoke(root, policy["reward_margin_penalty"])
            training_input = paired_training(train)
            context = coupled_training(trainer, original_loss, policy["reward_margin_penalty"])
        with context:
            model, tok, _ = trainer.train(str(base), training_input, validation_views, trained_path, hp)
    finally:
        trainer.build_model, trainer.evaluate, trainer.bt_loss = original_build, original_evaluate, original_loss
    verify_trained(trained_path, root, cfg)
    validation_report = validation_metrics(model)
    for style, measured in validation_report.items():
        measured["untrained_accuracy"] = initial_validation[style]["accuracy"]
    candidate = export(trained_path, FAMILY)
    candidate_bytes = candidate.read_bytes()
    check_candidate_tensor(candidate_bytes)
    del model
    gc.collect()
    foundation_spec = root/"foundation.json"
    write_json(foundation_spec, {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
        "foundation_version": "modernbert-base-lora16-v1", "tokenizer_hash": FOUNDATION["tokenizer.json"],
        "base_model": str(base), "r_max": 16, "num_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
        "targets": {key: list(value) for key, value in target_dims(cfg).items()}})
    compile_runtime(root, kit)
    for package in (candidate, root/"untrained.safetensors"):
        full_parser(root, package, foundation_spec)
    freeze = {"schema_version": 1, "family": FAMILY, "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "commit": os.environ["GITHUB_SHA"], "source_manifest_sha256": file_sha(root/"data/source-manifest.json"),
        "candidate_sha256": file_sha(candidate), "untrained_sha256": file_sha(root/"untrained.safetensors"),
        "validation": validation_report, "hyperparameters": asdict(hp), "audit_used_for_selection": False,
        "parent_run_id": PARENT_RUN, "parent_commit": PARENT_COMMIT, "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA,
        "training_originals": len(train), "training_views": len(training_views),
        "training_objective": args.objective,
        "auxiliary_training_pairs": source["auxiliary_training_pairs"], "reward_margin_penalty": policy["reward_margin_penalty"],
        "wider_training_pairs": source["wider_training_pairs"],
        "rank_growth_sha256": file_sha(root/"rank-growth.json"),
        "rank_growth_trained_sha256": file_sha(root/"rank-growth-trained.json"),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "safetensors", "numpy")}}
    if args.objective == "paired-worst-format":
        freeze["paired_objective_smoke_sha256"] = file_sha(root/"paired-objective-smoke.json")
    write_json(root/"training-freeze.json", freeze)
    print("Frozen licensed candidate and seed-matched untrained package. Independent audit was not scored.")
    # Preserve bytes and failure evidence first, then refuse to expose the audit.
    checked_validation = require_validation(validation_report, len(validation))
    write_json(root/"validation-ready.json", checked_validation)


if __name__ == "__main__":
    main()
