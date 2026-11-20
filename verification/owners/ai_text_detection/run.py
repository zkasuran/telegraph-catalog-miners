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
from development_contract import layout_training_pairs, repetition_gate, validate_locked
from window_contract import validate_plan, validate_augmented, expanded_training_pairs

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
    original_producer = json.loads((HERE / "recovery/locked-development-producer.json").read_text())
    locked_lineage = json.loads((HERE / "recovery/locked-development-lineage.json").read_text())
    source = json.loads((HERE / "sources.json").read_text())
    validate_locked((root / "development-generation.jsonl").read_bytes(), source, original_producer, locked_lineage)
    recipe = json.loads((HERE / "recipe.json").read_text())
    if recipe["source_manifest_sha256"] != sha((HERE / "sources.json").read_bytes()):
        raise ValueError("continuation recipe source manifest mismatch")
    if sha((root / "development-generation.jsonl").read_bytes()) != recipe["development_generation_sha256"]:
        raise ValueError("original source-fitted development bytes changed during continuation")
    splits = {s: [pair(r, "json" if int(r["group_id"][:2], 16) % 2 == 0 else "prose")
                  for r in original if r["split"] == s] for s in ("train", "validation")}
    plan = json.loads((HERE / "train-window-plan.json").read_text())
    validate_plan(plan, source, sha((HERE / "sources.json").read_bytes()))
    augmentation_raw = (root / "augmentation-generation.jsonl").read_bytes()
    augmentation_report = json.loads((root / "augmentation-generation-report.json").read_text())
    if (augmentation_report["run_id"] != int(os.environ["GITHUB_RUN_ID"]) or
            augmentation_report["commit"] != os.environ["GITHUB_SHA"] or
            augmentation_report["plan_sha256"] != sha((HERE / "train-window-plan.json").read_bytes())):
        raise ValueError("actual training augmentation run/commit/plan binding mismatch")
    augmented = validate_augmented(augmentation_raw, plan, source, augmentation_report)
    additional_plan_path = HERE / "additional-train-window-plan.json"
    additional_plan = json.loads(additional_plan_path.read_text())
    validate_plan(additional_plan, source, sha((HERE / "sources.json").read_bytes()))
    additional_raw = (root / "additional-augmentation-generation.jsonl").read_bytes()
    additional_report = json.loads((root / "additional-augmentation-generation-report.json").read_text())
    if (additional_report["run_id"] != int(os.environ["GITHUB_RUN_ID"]) or
            additional_report["commit"] != os.environ["GITHUB_SHA"] or
            additional_report["plan_sha256"] != sha(additional_plan_path.read_bytes())):
        raise ValueError("additional train-only generation run/commit/plan binding mismatch")
    augmented += validate_augmented(additional_raw, additional_plan, source, additional_report)
    validation_formats = {style: [pair(r, style) for r in original if r["split"] == "validation"] for style in FORMATS}
    validation_rows = [r for rows in validation_formats.values() for r in rows]
    if any(len(splits[s]) != 2 * COUNTS[s] for s in splits) or any(r["split"] == "test" for r in original):
        raise ValueError("development count or audit contamination")
    foundation, base, kit = prepare(root)
    subprocess.run(["go", "test", "-v", str(HERE / "stagecheck.go"), str(HERE / "stagecheck_test.go")],
                   cwd=kit, check=True, timeout=600)
    import torch
    import numpy as np
    from float_runtime import train_adapter as trainer
    from objective_contract import objective_policy
    from train_objective import smoke as objective_smoke
    objective_smoke(objective_policy(recipe), trainer, root)
    from train_coupled import smoke as coupled_smoke
    coupled_smoke(root)
    from train_margin import smoke as margin_smoke
    margin_smoke(root)
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import metadata, target_dims, tensor_names, write_package
    from tokenizers import Tokenizer
    from transformers import AutoConfig, AutoTokenizer
    torch.set_num_threads(4)
    node_check = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    node_check.no_padding()
    node_check.no_truncation()
    candidates, token_refusals = [], []
    grouped_views = {}
    for row in augmented:
        grouped_views.setdefault(row["view_id"], []).append(row)
    for view, rows in grouped_views.items():
        if any(len(node_check.encode(render(p["prompt"], p[side])).ids) > MAX_TOKENS
               for r in rows for style in FORMATS[:2] for p in [pair(r, style)] for side in ("chosen", "rejected")):
            token_refusals.append(view)
        else:
            candidates.extend(rows)
    splits["train"], expanded = expanded_training_pairs(original, candidates, bare_group_count=6)
    from compact_contract import compact_training
    splits["train"] = compact_training(splits["train"])
    expanded["training_answer_layout"] = "Opening/ending source quotes, four words each; original prompts and labels retained."
    expanded["full_context_refused_views"] = token_refusals
    expanded["augmentation_sha256"] = sha(augmentation_raw)
    expanded["additional_augmentation_sha256"] = sha(additional_raw)
    dump(root / "training-expansion-report.json", expanded)
    dump(root / "literal-repetition-gate.json", expanded["actual_repetition"])
    from supplement_contract import source_bundle, validate_generation
    from origin_contract import equal_byte_views
    from source_preferences import preference_policy, preference_rows
    preference_policy(recipe)
    bundle_sha, extra_sources, base_sources = source_bundle()
    supplement_report = json.loads((root / "supplement-generation-report.json").read_text())
    if supplement_report["run_id"] != int(os.environ["GITHUB_RUN_ID"]) or supplement_report["commit"] != os.environ["GITHUB_SHA"]:
        raise ValueError("supplement preference producer run/commit mismatch")
    supplement_views = validate_generation((root / "supplement-generation.jsonl").read_bytes(), supplement_report,
        extra_sources, base_sources, equal_byte_views(splits["train"]) + equal_byte_views(validation_formats["json"]))
    extra_preferences, preference_binding = preference_rows(supplement_views, splits["train"], validation_formats["json"])
    preference_binding.update(run_id=int(os.environ["GITHUB_RUN_ID"]), commit=os.environ["GITHUB_SHA"],
        source_manifest_sha256=bundle_sha, generation_sha256=sha((root / "supplement-generation.jsonl").read_bytes()))
    dump(root / "supplement-preference-contract.json", preference_binding)
    original_fitted = fit_baselines(splits["train"])
    fitted = fit_baselines(splits["train"] + extra_preferences)
    fitted["original_source_origin_length"] = original_fitted["source_origin_length"]
    dump(root / "frozen-baselines.json", fitted)
    baselines = {s: baseline_report(rows, fitted) for s, rows in splits.items()}
    baselines["supplement"] = baseline_report(extra_preferences, fitted)
    # Retain every preexisting answer-surface data gate. Source-origin length
    # additionally enters the full quality margin. Auxiliary inputs separately
    # require equal source bytes and a 50% source-length control.
    from common import pair_data_shortcuts_ok
    if not pair_data_shortcuts_ok(baselines):
        raise ValueError("expanded development shortcut exceeds unchanged 65% gate")
    validation_baselines = {style: baseline_report(rows, fitted) for style, rows in validation_formats.items()}
    hp = trainer.Hyperparams(max_length=MAX_TOKENS, batch_size=1, grad_accum=recipe["grad_accum"], epochs=recipe["epochs"], lr=recipe["lr"],
                            eval_batch=1, lora_rank=8, lora_alpha=32, gate_min_acc=0.80,
                            grad_checkpointing=False)
    original_gate = trainer.gate_dataset(splits["train"], hp)
    dump(root / "original-dataset-gate.json", original_gate)
    if not original_gate["ok"]:
        raise ValueError("unchanged original dataset gate refused the original 752 pairs")
    gate = trainer.gate_dataset(splits["train"] + extra_preferences, hp)
    dump(root / "dataset-gate.json", gate)
    if not gate["ok"]:
        raise ValueError("unchanged upstream dataset gate refused development corpus")
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    node = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    node.no_padding()
    node.no_truncation()
    for rows in [splits["train"], extra_preferences, validation_rows]:
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
    checkpoint = HERE / recipe.get("checkpoint_file", "recovery/original-checkpoint.safetensors")
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
        dump(root / "checkpoint-loaded.json", {"checkpoint_sha256": recipe["checkpoint_sha256"],
             "verified_learned_tensors": len(saved), "post_load_bits_identical_before_fitting": True,
             "parent_source_manifest_sha256": recipe["parent_source_manifest_sha256"]})
        return continued, tokenizer
    trainer.build_model = continuation_builder
    adapter = root / "adapter"
    started = time.monotonic()
    try:
        from train_formats import fit
        model, _ = fit(str(base), splits["train"], validation_formats, validation_baselines, adapter, hp, trainer, root)
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
    from supplement_contract import source_bundle
    source_hash, extra_sources, _ = source_bundle()
    training = {"family": FAMILY, "children": [{"intent": FAMILY + "_GENERAL", "pairs": len(splits["validation"]),
                 "dataset_version": source_hash, "accuracy": validation["pairwise_accuracy"], "held_out": True}],
                 "training_preferences": {"original": 752, "supplementary": 128, "total": 880},
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
    meta["x-extra-source-attribution"] = json.dumps(extra_sources["articles"], ensure_ascii=False)
    meta["x-base-source-manifest-sha256"] = sha((HERE / "sources.json").read_bytes())
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
    frozen["training_expansion"] = expanded
    frozen["training_batch_semantics"] = {"source_views_per_batch": 1, "literal_rows_per_batch": 2,
                                         "literal_rows_per_optimizer_update": hp.grad_accum}
    frozen["base_source_manifest_sha256"] = sha((HERE / "sources.json").read_bytes())
    frozen["supplement_generation_sha256"] = sha((root / "supplement-generation.jsonl").read_bytes())
    frozen["supplement_source_only_training_rows"] = 128
    frozen["supplement_preference_training"] = preference_binding
    dump(root / "frozen-candidate.json", frozen)
    if not selected_gate["passed"]:
        raise ValueError("all-format validation failed; preserve candidate and do not expose fresh audit")


if __name__ == "__main__":
    main()
