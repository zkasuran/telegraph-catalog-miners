# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Prepared remote-only keyword pilot. Missing approved data fails before models."""
import argparse
import gc
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import tarfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from gates import (
    SPLITS,
    check_pilot_counts,
    check_splits,
    fit_baselines,
    shortcut_report,
    structural_source_gate,
)
from heldout_freeze import verify_freeze
from prepare_data import MAX_TOKENS, answer, prompt
from scorer import render
from source import BASE, contains, parse, sha, verify_tree, write
from source import fetch as source_fetch
from terms_policy import licences_for

HERE = Path(__file__).resolve().parent


def remote_only():
    if os.environ.get("OWNER_SOURCE_ONLY") == "1" or os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_REPOSITORY") != "zkasuran/telegraph-catalog-miners":
        raise RuntimeError("Model training and scoring are restricted to the controller's bounded public GitHub CPU workflow")


def approved_data(directory):
    report = json.loads((directory / "approved-manifest.json").read_text())
    if report.get("family") != "KEYWORD_EXTRACTION" or report.get("max_tokens") != MAX_TOKENS:
        raise ValueError("approved dataset family/token identity mismatch")
    splits = {}
    for split in SPLITS:
        raw = (directory / (split + ".jsonl")).read_bytes()
        if len(raw) > 8 << 20 or sha(raw) != report["files"][split]["sha256"]:
            raise ValueError("approved split byte identity mismatch")
        splits[split] = [json.loads(line) for line in raw.decode().split("\n") if line]
    check_splits(splits)
    check_pilot_counts(splits)
    if not structural_source_gate(splits["train"])["ok"]:
        raise ValueError("unmodified primary structural source thresholds failed")
    freeze = directory / "heldout-freeze.json"
    if not freeze.is_file() or sha(freeze.read_bytes()) != report.get("heldout_freeze_sha256"):
        raise ValueError("actual frozen held-out source identity required")
    verify_freeze(splits, json.loads(freeze.read_text()))
    selection_hash = report.get("training_selection_sha256")
    if selection_hash:
        selection = directory / "training-selection.json"
        if not selection.is_file() or sha(selection.read_bytes()) != selection_hash:
            raise ValueError("training source-admission identity mismatch")
    if report.get("held_out_group_rule") not in {"topic-held-out plus DOI-author-donor components", "time-held-out plus DOI-author-donor components"}:
        raise ValueError("an explicit topic/time held-out group rule is required")
    return report, splits


def verify_sources(directory, splits):
    raw_inventory = (directory / "inventory.json").read_bytes()
    if len(raw_inventory) > 16 << 20:
        raise ValueError("source inventory ceiling exceeded")
    tree = json.loads(raw_inventory)
    verify_tree(tree)
    if tree["sha"] != "4ee79e98a7dbdcd3dd040be4b836e0843520c1fd":
        raise ValueError("source article tree differs from the verified eLife revision")
    entries = {e["path"]: e for e in tree["tree"]}
    total, verified = 0, {}
    fields = ("doi", "text", "title", "abstract", "author_keywords", "authors", "topics", "year", "licence", "licence_uri", "grant", "copyright")
    for rows in splits.values():
        for row in rows:
            if row.get("off_topic_donor"):
                donor = row.get("donor_source")
                if not isinstance(donor, dict) or donor["source_path"] not in entries:
                    raise ValueError("precision donor is not in the pinned source inventory")
                donor_path = donor["source_path"]
                if donor_path not in verified:
                    raw = source_fetch(BASE + donor_path)
                    total += len(raw)
                    if total > 128 << 20:
                        raise ValueError("approved source download budget exceeded")
                    verified[donor_path] = parse(raw, entries[donor_path])
                actual_donor = verified[donor_path]
                if actual_donor["sha256"] != donor["sha256"] or any(actual_donor[f] != donor[f] for f in fields):
                    raise ValueError("precision donor attribution or bytes differ from original")
                if row["rejected_term"] not in actual_donor["author_keywords"] or not contains(actual_donor["text"], row["rejected_term"]):
                    raise ValueError("precision donor phrase has no original annotation witness")
            source = row["source"]
            path = source["source_path"]
            if path not in entries or entries[path]["type"] != "blob":
                raise ValueError("approved source is not in the pinned inventory")
            if path not in verified:
                raw = source_fetch(BASE + path)
                total += len(raw)
                if total > 128 << 20:
                    raise ValueError("approved source download budget exceeded")
                verified[path] = parse(raw, entries[path])
            actual = verified[path]
            if actual["sha256"] != source["sha256"] or any(actual[f] != source[f] for f in fields):
                raise ValueError("approved source text, annotations or grants differ from original bytes")
            if row["prompt"] != prompt(actual):
                raise ValueError("approved prompt differs from complete source rendering")
            if row.get("off_topic_donor") and row["chosen_term"] not in actual["author_keywords"]:
                raise ValueError("precision chosen phrase requires its own original annotation")
            for side in ("chosen", "rejected"):
                term = row[side + "_term"]
                supported = not contains(actual["text"], term) if side == "rejected" and row.get("off_topic_donor") else contains(actual["text"], term)
                if not supported or row[side] != answer(term, "array"):
                    raise ValueError("reviewed term or answer differs from literal source-supported construction")
    return {"verified_originals": len(verified), "download_bytes": total, "inventory_sha256": sha(raw_inventory)}


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    config = json.loads((HERE / "config.json").read_text())
    data_report, splits = approved_data(args.data)
    write(args.root / "data-report.json", data_report)
    write(args.root / "source-verification.json", verify_sources(args.data, splits))
    frozen = fit_baselines(splits["train"])
    write(args.root / "frozen-baselines.json", frozen)
    # The untouched independent audit owns the test shortcut measurement once.
    # Train/validation gates cannot select a candidate using audit outcomes.
    shortcuts = {s: shortcut_report(splits[s], frozen) for s in ("train", "validation")}
    write(args.root / "shortcut-report.json", shortcuts)
    if any(v > config["maximum_data_shortcut"] for m in shortcuts.values() for v in m.values()):
        raise ValueError("frozen train-fitted shortcut exceeds 65%")
    # Shared immutable download/export interface, not a reused old adapter.
    import native as standard
    source = standard.SOURCES["kit"]
    archive = args.root / "kit.tar.gz"
    standard.fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + source["revision"], archive, source["sha256"], 32 << 20)
    with tarfile.open(archive) as tar:
        if sum(m.size for m in tar.getmembers()) > 128 << 20:
            raise ValueError("reference source archive ceiling exceeded")
        tar.extractall(args.root, filter="data")
    kit = args.root / ("telegraph-wasm-baseline-" + source["revision"])
    sys.path.insert(0, str(kit))
    from float_runtime import train_adapter as trainer
    hp = trainer.Hyperparams(max_length=MAX_TOKENS, batch_size=1, grad_accum=16, epochs=1, eval_batch=1,
                             lora_rank=8, lora_alpha=32, gate_min_acc=0.80, grad_checkpointing=False)
    gate = trainer.gate_dataset(splits["train"], hp)
    write(args.root / "dataset-gate.json", gate)
    if not gate["ok"]:
        raise ValueError("unmodified primary dataset gate failed before model preparation")
    foundation, base, kit = standard.prepare(args.root)
    import numpy as np
    import torch
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import (
        metadata,
        target_dims,
        tensor_names,
        write_package,
    )
    from tokenizers import Tokenizer
    from transformers import AutoConfig, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    node_tok = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    node_tok.no_truncation()
    node_tok.no_padding()
    checked = 0
    for rows in splits.values():
        for row in rows:
            for side in ("chosen", "rejected"):
                value = render(row["prompt"], row[side])
                ids = tok(value, truncation=False)["input_ids"]
                if ids != node_tok.encode(value).ids or len(ids) > MAX_TOKENS:
                    raise ValueError("intact input does not fit literal node token contract")
                checked += 1
    trainer.check_tokenizer(tok)
    write(args.root / "tokenizer-check.json", {"rendered_texts": checked, "max_tokens": MAX_TOKENS, "tokenizer_sha256": standard.FOUNDATION["tokenizer.json"]})
    torch.set_num_threads(config["cpu_threads"])
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    initial, _ = trainer.build_model(str(base), hp)
    initial_dir = args.root / "initial"
    initial.save_pretrained(initial_dir)
    trainer.save_reward_head(initial, initial_dir)
    del initial
    gc.collect()
    model, tok, _ = trainer.train(str(base), splits["train"], splits["validation"], args.root / "adapter", hp)
    (args.root / "adapter" / "tokenizer.json").write_bytes((foundation / "tokenizer.json").read_bytes())
    cfg = AutoConfig.from_pretrained(base, local_files_only=True)
    tensors, params = read_peft(args.root / "adapter", cfg)
    training = {"family": "KEYWORD_EXTRACTION", "child": "KEYWORD_EXTRACTION_GENERAL",
                "dataset_version": sha(json.dumps(data_report, sort_keys=True).encode()),
                "fit_pairs": len(splits["train"]), "validation_pairs": len(splits["validation"]),
                "audit_status": "unmeasured, no training accuracy record claimed",
                "source_scope": "Short English scientific abstract keyword centrality, AI-assisted constructed preferences, not universal correctness"}
    meta = metadata(cls="judgment", intent="KEYWORD_EXTRACTION", foundation_hash=standard.FOUNDATION["foundation_lora_fp32.onnx"],
                    foundation_version="modernbert-base-lora16-v1", rank=params["rank"], alpha=params["alpha"],
                    target_modules=params["target_modules"], max_tokens=MAX_TOKENS, training=None)
    meta["x-candidate-training"] = json.dumps(training, ensure_ascii=False)
    licences = licences_for(args.root)
    for key, value in licences.items():
        meta["x-terms-" + key] = value
    initial_tensors, initial_params = read_peft(initial_dir, cfg)
    if initial_params != params:
        raise ValueError("initialization and trained adapter layouts differ")
    initial_meta = dict(meta)
    initial_meta["x-candidate-training"] = json.dumps({"family": "KEYWORD_EXTRACTION", "seed": hp.seed,
        "scope": "Seed-matched untrained initialization, not a source-fit or qualified adapter"})
    initial_package = args.root / "KEYWORD_EXTRACTION_INITIAL.safetensors"
    write_package(initial_package, initial_meta, {n: initial_tensors[n] for n in tensor_names(cfg.num_hidden_layers)})
    package = args.root / "KEYWORD_EXTRACTION.safetensors"
    write_package(package, meta, {n: tensors[n] for n in tensor_names(cfg.num_hidden_layers)})
    del model
    gc.collect()
    spec = {"foundation_hash": meta["foundation_hash"], "foundation_version": meta["foundation_version"],
            "tokenizer_hash": meta["tokenizer_hash"], "base_model": str(base), "r_max": 16,
            "num_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
            "targets": {k: list(v) for k, v in target_dims(cfg).items()}}
    write(args.root / "foundation.json", spec)
    url = "https://devnode.telegraphprotocol.com/engine/v1/intents/KEYWORD_EXTRACTION_GENERAL"
    live = json.loads(source_fetch(url, 2 << 20))
    family_url = "https://devnode.telegraphprotocol.com/api/wasm?resolve=KEYWORD_EXTRACTION"
    parent = json.loads(source_fetch(family_url, 2 << 20))
    write(args.root / "class-snapshot.json", {"at": datetime.now(timezone.utc).isoformat(), "url": url, "child": live,
          "family_url": family_url, "parent": parent})
    checker = args.root / "keyword-packagecheck"
    executable = args.root / "adapterexec"
    subprocess.run(["go", "build", "-trimpath", "-o", str(checker), str(HERE / "packagecheck.go")], cwd=kit, check=True, timeout=600)
    subprocess.run(["go", "build", "-trimpath", "-o", str(executable), "./cmd/adapterexec"], cwd=kit, check=True, timeout=600)
    result = subprocess.run([str(checker), "--snapshot", str(args.root / "class-snapshot.json"), "--package", str(package),
                             "--foundation", str(args.root / "foundation.json"), "--out", str(args.root)], check=True, capture_output=True, text=True, timeout=60)
    manifest = json.loads(result.stdout)
    if manifest["sha256"] != sha(package.read_bytes()):
        raise ValueError("package static identity mismatch")
    write(args.root / "package-manifest.json", manifest)
    write(args.root / "candidate.json", {"family": "KEYWORD_EXTRACTION", "run_id": os.environ["GITHUB_RUN_ID"],
          "commit": os.environ["GITHUB_SHA"], "data": data_report, "package_sha256": manifest["sha256"],
          "initial_package_sha256": sha(initial_package.read_bytes()),
          "frozen_baselines_sha256": sha((args.root / "frozen-baselines.json").read_bytes()),
          "foundation_spec_sha256": sha((args.root / "foundation.json").read_bytes()),
          "hyperparameters": asdict(hp), "config_sha256": sha((HERE / "config.json").read_bytes()),
          "foundation": standard.FOUNDATION, "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "numpy", "safetensors")},
          "outcome": "needs independent audit", "node_qualification": "not attempted", "audit": "not measured by this training stage"})
    print("Frozen keyword candidate exported. Independent audit and literal node qualification are still required.")


if __name__ == "__main__":
    main()
