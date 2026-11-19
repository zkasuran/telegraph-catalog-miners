# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Train/validation freeze without inspecting the independent audit. CPU only."""
import argparse
import gc
import importlib.metadata
import os
import random
from dataclasses import asdict
from pathlib import Path

from baselines import validate_frozen
from contracts import FAMILY, FOUNDATION, HERE, check_config, file_sha, read_jsonl, remote_only, render, strict_json, write_json
from data import pair
from prepare_remote import compile_runtime, full_parser, prepare


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    root = parser.parse_args().root.resolve()
    config = check_config(strict_json((HERE/"config.json").read_bytes()))
    train, validation = [read_jsonl(root/"data"/(name+".jsonl")) for name in ("train", "validation")]
    source = strict_json((root/"data/source-manifest.json").read_bytes())
    for name in ("train", "validation", "audit"):
        if file_sha(root/"data"/(name+".jsonl")) != source["files"][name]["sha256"]:
            raise ValueError("frozen source data identity changed")
    validate_frozen(strict_json((HERE/"frozen-baselines-v2.json").read_bytes()), train)
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
    hp = trainer.Hyperparams(max_length=config["max_tokens"], batch_size=2, grad_accum=4, epochs=config["epochs"],
                            eval_batch=2, gate_min_acc=.80, grad_checkpointing=False, seed=config["seed"])
    gate = trainer.gate_dataset(train, hp)
    write_json(root/"upstream-dataset-gate.json", gate)
    if gate["ok"] is not True:
        raise ValueError("unmodified upstream dataset gate failed before model loading")
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
                    "pairs": len(train), "held_out": False}], "source_scope": "Four NDA clause claims, not performance compliance",
                    "preference_authorship": "Published document-level source labels and AI-assisted response templates, not human preference reviews"}
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
    initial_path = root/"untrained-adapter"
    initial.save_pretrained(initial_path)
    trainer.save_reward_head(initial, initial_path)
    export(initial_path, "untrained")
    del initial
    gc.collect()
    trained_path = root/"trained-adapter"
    model, tok, _ = trainer.train(str(base), train, validation, trained_path, hp)
    validation_report = trainer.evaluate(model, tok, validation, hp)
    candidate = export(trained_path, FAMILY)
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
    write_json(root/"training-freeze.json", {"schema_version": 1, "family": FAMILY, "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "commit": os.environ["GITHUB_SHA"], "source_manifest_sha256": file_sha(root/"data/source-manifest.json"),
        "candidate_sha256": file_sha(candidate), "untrained_sha256": file_sha(root/"untrained.safetensors"),
        "validation": validation_report, "hyperparameters": asdict(hp), "audit_used_for_selection": False,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "safetensors", "numpy")}})
    print("Frozen licensed candidate and seed-matched untrained package. Independent audit was not scored.")


if __name__ == "__main__":
    main()
