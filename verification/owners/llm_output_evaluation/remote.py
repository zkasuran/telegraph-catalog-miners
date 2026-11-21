# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Controller-only bounded CPU train/export and separate once-only audit."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import time

from data import FAMILY, RATINGS, control_scores, load_splits, prepare, render
from licensing import preserve, terms
from common import FOUNDATION
from token_contract import checked_tokens
from source_gate import digest, inventory, write
from wire import verify_qualification_render
from training_recipe import build_recipe, pair_bytes
from optimizer_contract import require_complete_accumulation


def remote_only():
    if (os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_REPOSITORY") != "zkasuran/telegraph-catalog-miners"
            or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
            or not os.environ.get("RUNNER_TEMP") or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"):
        raise RuntimeError("models run only in controller-owned standard public GitHub CPU workflows")


def reference(root):
    from reference import prepare as prepare_reference
    from types import SimpleNamespace
    foundation, base, kit = prepare_reference(root)
    return SimpleNamespace(FOUNDATION=FOUNDATION), foundation, base, kit


def identity(root):
    return {"owner_attempt": os.environ["OWNER_ATTEMPT"], "family": FAMILY,
            "github_run_id": os.environ["GITHUB_RUN_ID"], "github_commit": os.environ["GITHUB_SHA"],
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"), "repository": os.environ["GITHUB_REPOSITORY"],
            "node_schema_sha256": digest((root / "source/live-schema.json").read_bytes())}


def train(root):
    inventory(root / "source")
    preserve(root)
    wire = verify_qualification_render()
    live = json.loads((root / "source/live-schema.json").read_text())
    if live.get("class") != "judgment" or live.get("intent_id") != FAMILY + "_GENERAL":
        raise ValueError("live class mismatch")
    report = prepare(root / "source", root / "data")
    write(root / "result.json", {**identity(root), "phase": "data_gate", "data": report, "caller": wire,
                                 "outcome": "INCOMPLETE", "node_qualification": "not attempted"})
    if not report["training_allowed"]:
        raise ValueError("unchanged data quality gate failed")
    _, splits = load_splits(root / "data")
    fit_rows, recipe = build_recipe(splits["train"], os.environ.get("TRAINING_RECIPE", "json-original-v1"),
                                    report["files"]["train"]["sha256"], digest((root / "data/frozen-baseline.json").read_bytes()))
    recipe["optimizer_input_contract"] = require_complete_accumulation(len(fit_rows), 1, 16, recipe["epochs"])
    (root / "data/training_views.jsonl").write_bytes(pair_bytes(fit_rows))
    write(root / "training-recipe.json", recipe)
    helper, foundation, base, kit = reference(root)
    import numpy as np
    import torch
    from float_runtime import train_adapter as trainer
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import metadata, target_dims, tensor_names, write_package
    from transformers import AutoConfig, AutoTokenizer
    if torch.cuda.is_available():
        raise RuntimeError("GPU runtime refused")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    hp = trainer.Hyperparams(max_length=256, batch_size=1, grad_accum=16, lora_rank=8, epochs=recipe["epochs"],
                             eval_batch=1, gate_min_acc=0.80, grad_checkpointing=False)
    gate = trainer.gate_dataset(fit_rows, hp)
    write(root / "primary-data-gate.json", gate)
    if not gate["ok"]:
        raise ValueError("primary unmodified trainer data gate failed")
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    trainer.check_tokenizer(tok)
    from tokenizers import Tokenizer
    node = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    node.no_padding()
    node.no_truncation()
    checked = 0
    for split_rows in (fit_rows, splits["validation"], splits["audit"]):
        for row in split_rows:
            for side in ("chosen", "rejected"):
                text = render(row["prompt"], row[side])
                if text != trainer.render(row["prompt"], row[side]):
                    raise ValueError("literal primary trainer render mismatch")
                checked_tokens(text, tok, node, 256)
                checked += 1
    write(root / "token-visible.json", {"complete_vectors": checked, "tokenizer_hash": helper.FOUNDATION["tokenizer.json"], "native_node_parity": True})
    started = time.monotonic()
    model, tok, train_negatives = trainer.train(str(base), fit_rows, splits["validation"], root / "adapter", hp)
    preserve(root)
    metrics = trainer.evaluate(model, tok, splits["validation"], hp, train_negatives)
    write(root / "validation-result.json", metrics)
    if metrics["verdict"] != "PASS":
        raise ValueError("primary validation quality gate failed; checkpoint preserved")
    write(root / "training-timing.json", {"elapsed_seconds": time.monotonic() - started,
          "maximum_optimizer_steps": recipe["optimizer_steps_ceiling"], "maximum_epochs": recipe["epochs"],
          "training_recipe": recipe["recipe"], "example_presentations_ceiling": recipe["example_presentations_ceiling"],
          "audit_scored": False})
    cfg = AutoConfig.from_pretrained(base, local_files_only=True)
    tensors, params = read_peft(root / "adapter", cfg)
    meta = metadata(cls="judgment", intent=FAMILY, foundation_hash=helper.FOUNDATION["foundation_lora_fp32.onnx"],
                    foundation_version="modernbert-base-lora16-v1", rank=params["rank"], alpha=params["alpha"],
                    target_modules=params["target_modules"], max_tokens=256,
                    training={"family": FAMILY, "children": [{"intent": FAMILY + "_GENERAL",
                              "dataset_version": report["files"]["validation"]["sha256"],
                              "pairs": len(splits["validation"]), "accuracy": metrics["pairwise_accuracy"],
                               "held_out": True}], "trainer": "bounded CPU recipe " + recipe["recipe"] + ", validation selection only",
                              "backbone": "ModernBERT-base", "backbone_version": "8949b909ec900327062f0ebf497f51aef5e6f0c8"})
    meta.update(terms(root / "source"))
    meta["x-data-provenance"] = json.dumps(report, ensure_ascii=True, sort_keys=True)
    meta["x-training-recipe"] = json.dumps(recipe, ensure_ascii=True, sort_keys=True)
    path = root / (FAMILY + ".safetensors")
    write_package(path, meta, {n: tensors[n] for n in tensor_names(cfg.num_hidden_layers)})
    write(root / "foundation.json", {"foundation_hash": helper.FOUNDATION["foundation_lora_fp32.onnx"],
          "foundation_version": "modernbert-base-lora16-v1", "tokenizer_hash": helper.FOUNDATION["tokenizer.json"],
          "r_max": 16, "num_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
          "targets": {k: list(v) for k, v in target_dims(cfg).items()}})
    # Family is explicitly present in this kit. Do not override its class catalog.
    checked_package = subprocess.run(["go", "run", "-mod=readonly", "./cmd/adapterbuild", "lora", "--intent", FAMILY,
                    "--package", str(path), "--out", str(root), "--foundation", str(root / "foundation.json")],
                   cwd=kit, check=True, capture_output=True, timeout=600)
    write(root / "package-check.json", json.loads(checked_package.stdout))
    write(root / "selected.json", {**identity(root), "package_sha256": digest(path.read_bytes()),
          "selection": "bounded " + recipe["recipe"] + "; validation loss only, no audit scoring", "validation": metrics,
          "data": report, "caller": wire, "audit_scored": False,
          "training_recipe_sha256": digest((root / "training-recipe.json").read_bytes()),
          "baseline_sha256": digest((root / "data/frozen-baseline.json").read_bytes())})
    preserve(root)


def audit(root):
    from runtime_audit import run_audit
    run_audit(root, reference, identity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("train", "audit", "preserve", "serve"), required=True)
    args = parser.parse_args()
    if args.phase == "preserve":
        preserve(args.root)
    else:
        remote_only()
        args.root.mkdir(parents=True, exist_ok=True)
        if args.phase == "train":
            train(args.root)
        elif args.phase == "audit":
            audit(args.root)
        else:
            from serving import main as serve
            serve(args.root)
