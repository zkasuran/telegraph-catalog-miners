# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded remote training with the pinned Telegraph reference functions."""
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

from data import SOURCES, check_splits, decode_jsonl, digest, fetch, pair, write_json

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from prepare import FOUNDATION, RELEASE


def remote_only():
    if os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_REPOSITORY") != "zkasuran/telegraph-catalog-miners":
        raise RuntimeError("Training and scoring run only in the bounded GitHub workflow")


def read_splits(root):
    report = json.loads((root / "data-report.json").read_text())
    splits = {}
    for split in ("train", "validation", "test"):
        raw = (root / (split + ".jsonl")).read_bytes()
        if digest(raw) != report["files"][split]["sha256"]:
            raise ValueError("split hash mismatch")
        splits[split] = decode_jsonl(raw)
        for row in splits[split]:
            row["margin"] = 0.0
    check_splits(splits)
    return report, splits


def verdict(report):
    """All formats and both labels must pass, not just a favourable aggregate."""
    reasons = []
    if set(report["formats"]) != {"json", "prose", "bare_label"}:
        reasons.append("missing held-out response format")
    if not report["dataset_gate"]["ok"]:
        reasons.append("upstream dataset gate failed")
    for name, metrics in report["formats"].items():
        if metrics["pairwise_accuracy"] < 0.80 or not metrics["beats_length_baseline"]:
            reasons.append(name + " below accuracy or baseline gate")
        if metrics["pairwise_accuracy"] < report["untrained"][name]["pairwise_accuracy"] + 0.05:
            reasons.append(name + " did not improve the seed-matched untrained adapter by five points")
        if any(value < 0.70 for value in metrics["accuracy_by_label"].values()):
            reasons.append(name + " below per-label gate")
    return {"verdict": "FAIL" if reasons else "PASS", "failures": reasons}


def prepare(root):
    model, base = root / "foundation", root / "base"
    model.mkdir(exist_ok=True)
    base.mkdir(exist_ok=True)
    for name, sha in FOUNDATION.items():
        fetch(RELEASE + name, model / name, sha)
    # The published tokenizer is authoritative. Its vocabulary matches the HF
    # base but its serialization and configured padding/truncation differ.
    (base / "tokenizer.json").write_bytes((model / "tokenizer.json").read_bytes())
    spec = SOURCES["base"]
    for name, sha in spec["files"].items():
        fetch(f"https://huggingface.co/{spec['id']}/resolve/{spec['revision']}/{name}", base / name, sha)
    commit = SOURCES["kit"]["revision"]
    archive = root / "kit.tar.gz"
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + commit,
          archive, SOURCES["kit"]["sha256"], 32 << 20)
    with tarfile.open(archive) as source:
        if sum(m.size for m in source.getmembers()) > 128 << 20:
            raise ValueError("oversized reference kit")
        source.extractall(root, filter="data")
    kit = root / ("telegraph-wasm-baseline-" + commit)
    sys.path.insert(0, str(kit))
    os.environ["HF_HUB_OFFLINE"] = "1"
    return model, base, kit


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=["SENTIMENT_ANALYSIS", "CONTENT_MODERATION"])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    data_report, splits = read_splits(root / "data")
    foundation, base, kit = prepare(root)
    import numpy as np
    import torch
    from float_runtime import train_adapter as trainer
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import (
        metadata,
        target_dims,
        tensor_names,
        write_package,
    )
    from tokenizers import Tokenizer
    from transformers import AutoConfig, AutoTokenizer

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    hp = trainer.Hyperparams(max_length=256, batch_size=2, grad_accum=8, epochs=3,
                            eval_batch=4, gate_min_acc=0.80, grad_checkpointing=False)
    data_gate = trainer.gate_dataset(splits["train"], hp)
    write_json(root / "dataset-gate.json", data_gate)
    if not data_gate["ok"]:
        write_json(root / "result.json", {"verdict": "FAIL", "phase": "dataset", "dataset_gate": data_gate})
        raise RuntimeError("Dataset gate failed before loading any model")
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    node_tok = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    node_tok.no_padding()
    node_tok.enable_truncation(max_length=256)
    test_formats = {}
    for style, name in enumerate(("json", "prose", "bare_label")):
        rows = []
        for original in splits["test"]:
            source = {**original, "text": json.loads(original["prompt"])["text"]}
            rows.append({**pair(args.family, source, style), "margin": 0.0})
        test_formats[name] = rows
    checked = 0
    for rows in list(splits.values()) + list(test_formats.values()):
        for row in rows:
            for side in ("chosen", "rejected"):
                text = trainer.render(row["prompt"], row[side])
                ids = tok(text, truncation=True, max_length=256)["input_ids"]
                if ids != node_tok.encode(text).ids:
                    raise ValueError("training tokenizer differs from node tokenizer")
                # Never train or score a pair whose differing claim was truncated.
                if len(tok(text, truncation=False)["input_ids"]) > 256:
                    raise ValueError("source pair exceeds training token ceiling")
                checked += 1
    trainer.check_tokenizer(tok)
    write_json(root / "tokenizer-check.json", {"rendered_texts": checked, "matches_node": True,
                                             "tokenizer_sha256": FOUNDATION["tokenizer.json"]})

    def evaluate(model):
        result = {}
        for name, rows in test_formats.items():
            metrics = trainer.evaluate(model, tok, rows, hp)
            metrics["accuracy_by_label"] = {
                label: trainer.evaluate(model, tok, [r for r in rows if r["label"] == label], hp)["pairwise_accuracy"]
                for label in sorted({r["label"] for r in rows})}
            result[name] = metrics
        return result

    # Same initialization and backbone. Test is never used for epoch selection.
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    initial, _ = trainer.build_model(str(base), hp)
    untrained = evaluate(initial)
    del initial
    gc.collect()
    started = time.monotonic()
    adapter = root / "adapter"
    model, tok, _ = trainer.train(str(base), splits["train"], splits["validation"], adapter, hp)
    trained = evaluate(model)
    # Serialization can change between tokenizers releases. Preserve the actual
    # authoritative tokenizer bytes after the token-for-token conformance check.
    (adapter / "tokenizer.json").write_bytes((foundation / "tokenizer.json").read_bytes())
    report = {"family": args.family, "dataset_gate": data_gate, "sources": data_report,
              "hyperparameters": asdict(hp), "untrained": untrained, "formats": trained,
              "elapsed_training_and_evaluation_seconds": round(time.monotonic() - started, 2),
              "versions": {n: importlib.metadata.version(n) for n in
                           ("torch", "transformers", "peft", "numpy", "safetensors", "huggingface-hub")},
              "run_id": os.environ["GITHUB_RUN_ID"], "commit": os.environ["GITHUB_SHA"],
              "scope": SOURCES[args.family]["scope"], "node_qualification": "not attempted"}
    report.update(verdict(report))
    write_json(root / "result.json", report)
    print(json.dumps({"family": args.family, **verdict(report), "formats": trained}))
    if report["verdict"] != "PASS":
        return 1

    cfg = AutoConfig.from_pretrained(base, local_files_only=True)
    tensors, params = read_peft(adapter, cfg)
    training = {"family": args.family, "children": [{"intent": args.family + "_GENERAL",
                "dataset_version": digest((root / "data" / "data-report.json").read_bytes()),
                "pairs": len(splits["test"]), "accuracy": trained["json"]["pairwise_accuracy"], "held_out": True}],
                "base_model": SOURCES["base"], "source_scope": report["scope"],
                "preference_authorship": "Programmatically constructed from published source labels; AI-assisted code."}
    meta = metadata(cls="judgment", intent=args.family, foundation_hash=FOUNDATION["foundation_lora_fp32.onnx"],
                    foundation_version="modernbert-base-lora16-v1", rank=params["rank"], alpha=params["alpha"],
                    target_modules=params["target_modules"], max_tokens=256, training=training)
    licences = json.loads((HERE / "licences.json").read_text())
    meta["x-upstream-notice"] = licences["notice"]
    meta["x-upstream-apache-2.0"] = licences["apache_2_0"]
    meta["x-upstream-mit"] = licences["kit_mit"]
    meta["x-licence"] = licences["sand"]
    package = root / (args.family + ".safetensors")
    write_package(package, meta, {n: tensors[n] for n in tensor_names(cfg.num_hidden_layers)})
    del model
    gc.collect()
    executable = root / "adapterexec"
    subprocess.run(["go", "build", "-trimpath", "-o", str(executable), "./cmd/adapterexec"], cwd=kit, check=True, timeout=600)
    checker = root / "adapterbuild"
    subprocess.run(["go", "build", "-trimpath", "-o", str(checker), "./cmd/adapterbuild"], cwd=kit, check=True, timeout=600)
    foundation_spec = {"foundation_hash": meta["foundation_hash"], "foundation_version": meta["foundation_version"],
                       "tokenizer_hash": meta["tokenizer_hash"], "base_model": str(base), "r_max": 16,
                       "num_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
                       "targets": {k: list(v) for k, v in target_dims(cfg).items()}}
    write_json(root / "foundation.json", foundation_spec)
    checked = subprocess.run([str(checker), "lora", "--intent", args.family, "--package", str(package),
                              "--class", "judgment", "--foundation", str(root / "foundation.json"),
                              "--out", str(root)], check=True, capture_output=True, text=True, timeout=60)
    manifest = json.loads(checked.stdout)
    if manifest["sha256"] != digest(package.read_bytes()):
        raise ValueError("node parser package hash mismatch")
    write_json(root / "package-manifest.json", manifest)
    subprocess.run([sys.executable, str(HERE / "runtime_check.py"), "--root", str(root)], check=True, timeout=3600)
    return 0


if __name__ == "__main__":
    sys.exit(main())
