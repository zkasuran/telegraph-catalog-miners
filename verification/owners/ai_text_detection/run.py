# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only clean LoRA training. Audit rows are never opened here."""
import argparse
import gc
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import tarfile
import time
from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone

from common import (COUNTS, FAMILY, HERE, MAX_TOKENS, SCOPE, baseline_report, dump,
                    fetch, fit_baselines, lines, pair, remote_only, render, sha, validation_gate, FORMATS)
from checkpoint_contract import check_checkpoint

FOUNDATION = {
    "foundation_lora_fp32.onnx": "0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6",
    "foundation_engine.wasm": "77643d6e1176ab8744801cef93ccb844d0136ed32b8daa865d541f37d13292de",
    "tokenizer.json": "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456",
}
RELEASE = "https://github.com/telegraphprotocol/telegraph-foundation/releases/download/modernbert-base-lora16-v1/"


def prepare(root):
    pins = json.loads((HERE / "model-pins.json").read_text())
    foundation, base = root / "foundation", root / "base"
    foundation.mkdir(exist_ok=True)
    base.mkdir(exist_ok=True)
    for name, expected in FOUNDATION.items():
        fetch(RELEASE + name, foundation / name, expected, 900 << 20)
    for name, expected in pins["base"]["files"].items():
        s = pins["base"]
        fetch(f"https://huggingface.co/{s['id']}/resolve/{s['revision']}/{name}", base / name, expected, 900 << 20)
    (base / "tokenizer.json").write_bytes((foundation / "tokenizer.json").read_bytes())
    archive = root / "kit.tar.gz"
    commit = pins["kit"]["revision"]
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + commit,
          archive, pins["kit"]["sha256"], 32 << 20)
    with tarfile.open(archive) as source:
        if sum(m.size for m in source.getmembers()) > 128 << 20:
            raise ValueError("reference source archive ceiling")
        source.extractall(root, filter="data")
    kit = root / ("telegraph-wasm-baseline-" + commit)
    sys.path.insert(0, str(kit))
    os.environ["HF_HUB_OFFLINE"] = "1"
    return foundation, base, kit


def main():
    remote_only()
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    root = p.parse_args().root.resolve()
    generation_report = json.loads((root / "development-generation-report.json").read_text())
    if (generation_report["output_sha256"] != sha((root / "development-generation.jsonl").read_bytes()) or
            generation_report["source_manifest_sha256"] != sha((HERE / "sources.json").read_bytes()) or
            generation_report["commit"] != os.environ["GITHUB_SHA"] or
            generation_report["run_id"] != int(os.environ["GITHUB_RUN_ID"])):
        raise ValueError("development generation provenance hash mismatch")
    original = lines((root / "development-generation.jsonl").read_bytes())
    recipe = json.loads((HERE / "recipe.json").read_text())
    if recipe["source_manifest_sha256"] != sha((HERE / "sources.json").read_bytes()):
        raise ValueError("continuation recipe source manifest mismatch")
    if sha((root / "development-generation.jsonl").read_bytes()) != recipe["development_generation_sha256"]:
        raise ValueError("original source-fitted development bytes changed during continuation")
    # Four whole article groups expose both origins in the literal bare format.
    # Eight bare answers keep each repeated opposite label below the unchanged
    # upstream 2% repetition ceiling. Other groups retain JSON/prose templates.
    bare_groups = set(sorted({r["group_id"] for r in original if r["split"] == "train"})[:4])
    splits = {s: [pair(r, "json" if int(r["group_id"][:2], 16) % 2 == 0 else "prose")
                  for r in original if r["split"] == s] for s in ("train", "validation")}
    splits["train"] = [pair(r, "bare_label" if r["group_id"] in bare_groups else
                            "json" if int(r["group_id"][:2], 16) % 2 == 0 else "prose")
                       for r in original if r["split"] == "train"]
    validation_formats = {style: [pair(r, style) for r in original if r["split"] == "validation"] for style in FORMATS}
    validation_rows = [r for rows in validation_formats.values() for r in rows]
    if any(len(splits[s]) != 2 * COUNTS[s] for s in splits) or any(r["split"] == "test" for r in original):
        raise ValueError("development count or audit contamination")
    fitted = fit_baselines(splits["train"])
    dump(root / "frozen-baselines.json", fitted)
    baselines = {s: baseline_report(rows, fitted) for s, rows in splits.items()}
    if any(v > 0.65 for values in baselines.values() for v in values.values()):
        raise ValueError("development shortcut exceeds 65%")
    foundation, base, kit = prepare(root)
    subprocess.run(["go", "test", "-v", str(HERE / "stagecheck.go"), str(HERE / "stagecheck_test.go")],
                   cwd=kit, check=True, timeout=600)
    import torch
    import numpy as np
    from float_runtime import train_adapter as trainer
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import metadata, target_dims, tensor_names, write_package
    from tokenizers import Tokenizer
    from transformers import AutoConfig, AutoTokenizer
    torch.set_num_threads(4)
    hp = trainer.Hyperparams(max_length=MAX_TOKENS, batch_size=1, grad_accum=recipe["grad_accum"], epochs=recipe["epochs"], lr=recipe["lr"],
                            eval_batch=1, lora_rank=8, lora_alpha=32, gate_min_acc=0.80,
                            grad_checkpointing=False)
    gate = trainer.gate_dataset(splits["train"], hp)
    dump(root / "dataset-gate.json", gate)
    if not gate["ok"]:
        raise ValueError("unchanged upstream dataset gate refused development corpus")
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    node = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    node.no_padding()
    node.no_truncation()
    for rows in [splits["train"], validation_rows]:
        for r in rows:
            for side in ("chosen", "rejected"):
                text = render(r["prompt"], r[side])
                if trainer.render(r["prompt"], r[side]) != text:
                    raise ValueError("literal node/trainer render mismatch")
                ids = tok(text, truncation=False)["input_ids"]
                if len(ids) > MAX_TOKENS or ids != node.encode(text).ids:
                    raise ValueError("full-context tokenizer identity or ceiling mismatch")
    trainer.check_tokenizer(tok)
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    initial, _ = trainer.build_model(str(base), hp)
    initial_dir = root / "initial"
    initial.save_pretrained(initial_dir)
    trainer.save_reward_head(initial, initial_dir)
    del initial
    gc.collect()
    # Continue exactly the previously licensed LoRA/head. The original optimizer
    # was not exported, so this is explicitly a fresh bounded optimizer, not an
    # invented optimizer-state resume. Backbone weights stay frozen.
    checkpoint = HERE / "recovery/original-checkpoint.safetensors"
    checkpoint_binding = check_checkpoint(checkpoint.read_bytes(), recipe["checkpoint_sha256"], recipe["parent_source_manifest_sha256"])
    dump(root / "checkpoint-continuation.json", checkpoint_binding)
    build = trainer.build_model
    def continuation_builder(base_path, parameters):
        from safetensors.torch import load_file
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        continued, tokenizer = build(base_path, parameters)
        saved = load_file(str(checkpoint), device="cpu")
        expected = get_peft_model_state_dict(continued, save_embedding_layers=False)
        if set(saved) != set(expected) or any(saved[k].shape != expected[k].shape or not torch.isfinite(saved[k]).all() for k in saved):
            raise ValueError("checkpoint learned tensor keys/shapes/finiteness mismatch")
        set_peft_model_state_dict(continued, saved, adapter_name="default")
        actual = get_peft_model_state_dict(continued, save_embedding_layers=False)
        if any(not torch.equal(saved[k], actual[k]) for k in saved):
            raise ValueError("warm-started learned tensor differs from exact saved checkpoint")
        return continued, tokenizer
    trainer.build_model = continuation_builder
    adapter = root / "adapter"
    started = time.monotonic()
    try:
        model, _, _ = trainer.train(str(base), splits["train"], validation_rows, adapter, hp)
    finally:
        trainer.build_model = build
    (adapter / "tokenizer.json").write_bytes((foundation / "tokenizer.json").read_bytes())
    validation_results = {}
    for style, rows in validation_formats.items():
        m = trainer.evaluate(model, tok, rows, hp)
        m["accuracy_by_label"] = {label: trainer.evaluate(model, tok, [r for r in rows if r["label"] == label], hp)["pairwise_accuracy"]
                                  for label in ("Human", "AI-generated")}
        validation_results[style] = m
    validation_baselines = {style: baseline_report(rows, fitted) for style, rows in validation_formats.items()}
    selected_gate = validation_gate(validation_results, validation_baselines)
    dump(root / "validation-result.json", {"formats": validation_results, "baselines": validation_baselines, "gate": selected_gate,
                                          "independent_article_groups": COUNTS["validation"], "audit_used": False})
    validation = validation_results["json"]
    cfg = AutoConfig.from_pretrained(base, local_files_only=True)
    tensors, params = read_peft(adapter, cfg)
    source_hash = sha((HERE / "sources.json").read_bytes())
    training = {"family": FAMILY, "children": [{"intent": FAMILY + "_GENERAL", "pairs": len(splits["validation"]),
                "dataset_version": source_hash, "accuracy": validation["pairwise_accuracy"], "held_out": True}],
                "source_scope": SCOPE, "selection": "Validation only; audit not opened.",
                "preference_authorship": "Constructed by AI-assisted code from published controls and disclosed synthetic origin. Not human-reviewed."}
    meta = metadata(cls="judgment", intent=FAMILY, foundation_hash=FOUNDATION["foundation_lora_fp32.onnx"],
                    foundation_version="modernbert-base-lora16-v1", rank=params["rank"], alpha=params["alpha"],
                    target_modules=params["target_modules"], max_tokens=MAX_TOKENS, training=training)
    source = json.loads((HERE / "sources.json").read_text())
    # Upstream terms precede our additions. Source terms are not overwritten.
    for name, item in source["terms"].items():
        meta["x-terms-" + name] = item["text"]
    meta["x-upstream-notice"] = (HERE / "NOTICE").read_text()
    meta["x-source-manifest-sha256"] = source_hash
    meta["x-checkpoint-lineage"] = json.dumps(recipe, sort_keys=True)
    meta["x-source-attribution"] = json.dumps([{k: r[k] for k in
        ("source_id", "authors", "url", "licence", "licence_url", "changes", "sha256")}
        for rows in source["source_report"]["articles"].values() for r in rows], ensure_ascii=False)
    package = root / (FAMILY + ".safetensors")
    write_package(package, meta, {n: tensors[n] for n in tensor_names(cfg.num_hidden_layers)})
    initial_tensors, _ = read_peft(initial_dir, cfg)
    initial_meta = {**meta, "x-control": "Seed-matched untrained LoRA/head control. Its reference-engine result, not this shared metadata, measures baseline quality."}
    write_package(root / "UNTRAINED.safetensors", initial_meta,
                  {n: initial_tensors[n] for n in tensor_names(cfg.num_hidden_layers)})
    del initial_tensors
    dump(root / "foundation.json", {"foundation_hash": meta["foundation_hash"],
         "foundation_version": meta["foundation_version"], "tokenizer_hash": meta["tokenizer_hash"],
         "base_model": str(base), "r_max": 16, "num_layers": cfg.num_hidden_layers,
         "hidden_size": cfg.hidden_size, "targets": {k: list(v) for k, v in target_dims(cfg).items()}})
    del model, tensors
    gc.collect()
    for name in ("adapterexec",):
        subprocess.run(["go", "build", "-trimpath", "-o", str(root / name), "./cmd/" + name],
                       cwd=kit, check=True, timeout=600)
    # Use the unchanged CheckLoRA with a source-bound fresh live class registry,
    # not a family override or an incomplete compiled offline catalog.
    subprocess.run(["go", "build", "-trimpath", "-o", str(root / "stagecheck"), str(HERE / "stagecheck.go")],
                   cwd=kit, check=True, timeout=600)
    snapshot = {"schema": 1, "fetched_at": datetime.now(timezone.utc).isoformat()}
    for name, url in (("family", "https://devnode.telegraphprotocol.com/api/wasm?resolve=" + FAMILY),
                      ("child", "https://devnode.telegraphprotocol.com/engine/v1/intents/" + FAMILY + "_GENERAL")):
        raw = fetch(url, root / (name + "-class-raw.json"), ceiling=2 << 20)
        snapshot[name] = {"url": url, "raw": raw.decode()}
    dump(root / "class-snapshot.json", snapshot)
    checked = subprocess.run([str(root / "stagecheck"), "--intent", FAMILY, "--child", FAMILY + "_GENERAL",
                              "--expected-class", "judgment", "--package", str(package), "--sha256", sha(package.read_bytes()),
                              "--class-snapshot", str(root / "class-snapshot.json"),
                              "--snapshot-sha256", sha((root / "class-snapshot.json").read_bytes()),
                              "--foundation", str(root / "foundation.json"),
                              "--foundation-sha256", sha((root / "foundation.json").read_bytes()), "--out", str(root)],
                             capture_output=True, text=True, check=True, timeout=120)
    manifest = json.loads(checked.stdout)
    if manifest["sha256"] != sha(package.read_bytes()):
        raise ValueError("full package checker identity mismatch")
    dump(root / "package-manifest.json", manifest)
    frozen = {"family": FAMILY, "package_sha256": manifest["sha256"], "package_keccak": manifest["keccak256"],
              "source_manifest_sha256": source_hash, "development_sha256": sha((root / "development-generation.jsonl").read_bytes()),
              "baseline_sha256": sha((root / "frozen-baselines.json").read_bytes()), "validation": validation,
              "hyperparameters": asdict(hp), "audit_used_for_selection": False,
              "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
              "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "numpy", "safetensors")},
              "elapsed_training_seconds": time.monotonic() - started, "full_parser_passed": True,
              "validation_formats": validation_results, "validation_gate": selected_gate,
              "checkpoint_lineage": recipe, "independent_validation_groups": COUNTS["validation"]}
    dump(root / "frozen-candidate.json", frozen)
    if not selected_gate["passed"]:
        raise ValueError("all-format validation failed; preserve candidate and do not expose fresh audit")


if __name__ == "__main__":
    main()
