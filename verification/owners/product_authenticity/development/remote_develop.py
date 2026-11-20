# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded real CPU development training. It cannot emit registration readiness."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import time
import urllib.request

from data_contract import CHILD, FAMILY, FORMATS, build, digest, render
from runtime_contract import execute, remote_only

HERE = Path(__file__).resolve().parent


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def fetch(url, path, expected, ceiling):
    if path.exists() and digest(path.read_bytes()) == expected:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "ProductAuthenticityCPUDevelopment/1.0"})
    total, h = 0, hashlib.sha256()
    deadline = time.monotonic() + 600
    with urllib.request.urlopen(req, timeout=60) as response, path.open("wb") as output:
        while chunk := response.read(1 << 20):
            total += len(chunk)
            if total > ceiling or time.monotonic() > deadline:
                raise ValueError("download byte/time ceiling")
            h.update(chunk)
            output.write(chunk)
    if h.hexdigest() != expected:
        raise ValueError("download hash mismatch: " + path.name)


def class_snapshot(root):
    records = {}
    for name, url in (("family", "https://devnode.telegraphprotocol.com/api/wasm?resolve=" + FAMILY),
                      ("child", "https://devnode.telegraphprotocol.com/engine/v1/intents/" + CHILD)):
        req = urllib.request.Request(url, headers={"User-Agent": "ProductAuthenticitySourceResearch/2.0"})
        with urllib.request.urlopen(req, timeout=25) as response:
            raw = response.read((1 << 20) + 1)
        if len(raw) > 1 << 20:
            raise ValueError("class metadata ceiling")
        records[name] = {"url": url, "raw": raw.decode()}
    snapshot = {"schema": 1, "fetched_at": datetime.now(timezone.utc).isoformat(), **records}
    write(root / "class-snapshot.json", snapshot)
    return digest((root / "class-snapshot.json").read_bytes())


def metrics(scores, rows):
    values = [float(x) for x in scores]
    if len(values) != 2 * len(rows) or any(not math.isfinite(x) for x in values):
        raise ValueError("actual score count or finiteness")
    wins = [1.0 if values[2*i] > values[2*i+1] else .5 if values[2*i] == values[2*i+1] else 0.0 for i in range(len(rows))]
    return {"accuracy": sum(wins) / len(rows), "pairs": len(rows),
            "per_label": {label: sum(w for r, w in zip(rows, wins) if r["label"] == label) / sum(r["label"] == label for r in rows) for label in sorted({r["label"] for r in rows})},
            "per_kind": {kind: sum(w for r, w in zip(rows, wins) if r["kind"] == kind) / sum(r["kind"] == kind for r in rows) for kind in sorted({r["kind"] for r in rows})}}


def frozen_answer_baseline(np, train, validation, root):
    import re
    token = re.compile(r"[a-z0-9]+")
    def features(value):
        words = token.findall(value.lower())
        return words + [a + " " + b for a, b in zip(words, words[1:])]
    counts = {}
    for row in train:
        for side in ("chosen", "rejected"):
            for word in set(features(row[side])):
                counts[word] = counts.get(word, 0) + 1
    vocab = {word: i for i, word in enumerate(sorted(counts, key=lambda w: (-counts[w], w))[:5000])}
    def vec(value):
        out = np.zeros(len(vocab), dtype=np.float32)
        for word in features(value):
            if word in vocab:
                out[vocab[word]] += 1
        norm = np.linalg.norm(out)
        return out / max(float(norm), 1.0)
    x = np.array([vec(r[s]) for r in train for s in ("chosen", "rejected")], dtype=np.float32)
    y = np.array([v for _ in train for v in (1.0, 0.0)], dtype=np.float32)
    weights, bias = np.zeros(len(vocab), dtype=np.float32), np.float32(0)
    for _ in range(60):
        p = 1 / (1 + np.exp(-np.clip(x @ weights + bias, -20, 20)))
        error = (p - y) / len(y)
        weights -= np.float32(.5) * (x.T @ error + np.float32(1e-4) * weights)
        bias -= np.float32(.5) * error.sum()
    # Immutable train-fitted bits. Reproduction never requires fitting again.
    frozen = {"vocabulary": vocab, "weight_bits": weights.astype("<f4").view("<u4").tolist(),
              "bias_bits": np.array([bias], dtype="<f4").view("<u4").tolist()[0], "fitted_on": "train_only",
              "train_sha256": digest((root / "data/train.jsonl").read_bytes()), "features": "answer-only unigrams and word bigrams"}
    write(root / "frozen-answer-only-baseline.json", frozen)
    by_format = {}
    for style in FORMATS:
        rows = [r for r in validation if r["format"] == style]
        scores = [float(vec(r[s]) @ weights + bias) for r in rows for s in ("chosen", "rejected")]
        by_format[style] = metrics(scores, rows)
    return {"formats": by_format, "frozen_coefficients_sha256": digest((root / "frozen-answer-only-baseline.json").read_bytes()), "fit_source": "train_only"}


def main(data_builder=build, max_tokens=256, control_hook=None, development_mode="develop"):
    remote_only()
    if max_tokens not in (256, 512) or development_mode not in ("develop", "develop_broad"):
        raise ValueError("unsupported bounded development configuration")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    source_report, splits = data_builder(root / "data")
    write(root / "development-report.json", {"schema_version": 1, "status": "DEVELOPMENT_INCOMPLETE", "family": FAMILY,
          "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"], "data": source_report,
          "audit_pairs": 0, "registration_ready": False})
    sources = json.loads((HERE / "model-sources.json").read_text())
    archive = root / "kit.tar.gz"
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + sources["kit"]["revision"], archive, sources["kit"]["sha256"], 32 << 20)
    with tarfile.open(archive) as source:
        if sum(m.size for m in source.getmembers()) > 128 << 20:
            raise ValueError("source kit expansion ceiling")
        source.extractall(root, filter="data")
    kit = root / ("telegraph-wasm-baseline-" + sources["kit"]["revision"])
    subprocess.run(["go", "test", "-mod=readonly", str(HERE / "packagecheck.go"), str(HERE / "packagecheck_test.go")], cwd=kit, check=True, timeout=600)
    for name, source in (("packagecheck", str(HERE / "packagecheck.go")), ("adapterexec", "./cmd/adapterexec")):
        subprocess.run(["go", "build", "-mod=readonly", "-trimpath", "-o", str(root / name), source], cwd=kit, check=True, timeout=600)
    snapshot_sha = class_snapshot(root)
    subprocess.run([str(root / "packagecheck"), "--intent", FAMILY, "--child", CHILD, "--class-snapshot", str(root / "class-snapshot.json"), "--snapshot-sha256", snapshot_sha, "--preflight"], check=True, timeout=60)
    sys.path.insert(0, str(kit))
    import numpy as np
    import torch
    from float_runtime import train_adapter as trainer
    from licensing import preserve_artifact
    hp = trainer.Hyperparams(max_length=max_tokens, batch_size=1, grad_accum=8, epochs=3, eval_batch=2,
                             lora_rank=8, lora_alpha=16, lora_dropout=0.0, grad_checkpointing=False,
                             gate_min_acc=.8, early_stopping=False, lr=4e-5)
    gate = trainer.gate_dataset(splits["train"], hp)
    write(root / "dataset-gate.json", gate)
    if not gate["ok"]:
        raise ValueError("unchanged primary dataset shortcut gate failed before model loading")
    if torch.cuda.is_available():
        raise RuntimeError("paid/GPU runtime refused")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    baseline = frozen_answer_baseline(np, splits["train"], splits["validation"], root)
    foundation, base = root / "foundation", root / "base"
    foundation.mkdir(exist_ok=True)
    base.mkdir(exist_ok=True)
    for name, expected in sources["foundation"].items():
        fetch(sources["foundation_release"] + name, foundation / name, expected, 2 << 30)
    for name, expected in sources["base"]["files"].items():
        fetch("https://huggingface.co/" + sources["base"]["id"] + "/resolve/" + sources["base"]["revision"] + "/" + name, base / name, expected, 2 << 30)
    (base / "tokenizer.json").write_bytes((foundation / "tokenizer.json").read_bytes())
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoConfig, AutoTokenizer
    from tokenizers import Tokenizer
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if not trainer.TOKENIZER_VECTORS.exists():
        raise ValueError("primary tokenizer vectors missing")
    trainer.check_tokenizer(tok)
    node = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    node.no_padding()
    node.no_truncation()
    for rows in splits.values():
        for row in rows:
            for side in ("chosen", "rejected"):
                value = render(row["prompt"], row[side])
                if tok(value, truncation=False)["input_ids"] != node.encode(value).ids or len(node.encode(value).ids) > max_tokens:
                    raise ValueError("literal tokenizer mismatch or truncated contrasting claim")
    extra_controls = control_hook(splits, node, root) if control_hook else {}
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    model, tok = trainer.build_model(str(base), hp)
    def save_checkpoint(destination):
        destination.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(destination)
        trainer.save_reward_head(model, destination)
        original_head = torch.load(destination / "reward_head.pt", map_location="cpu", weights_only=True)
        for path in [*destination.glob("*.safetensors"), destination / "reward_head.pt"]:
            preserve_artifact(path)
        restored_head = torch.load(destination / "reward_head.pt", map_location="cpu", weights_only=True)
        if original_head.keys() != restored_head.keys() or any(not torch.equal(original_head[k], restored_head[k]) for k in original_head):
            raise ValueError("checkpoint licence record changed head tensors")
    save_checkpoint(root / "initial")
    def evaluate(rows):
        model.eval()
        actual = []
        with torch.no_grad():
            for row in rows:
                for side in ("chosen", "rejected"):
                    encoded = tok(render(row["prompt"], row[side]), truncation=False, return_tensors="pt")
                    actual.append(trainer._reward(model, encoded["input_ids"], encoded["attention_mask"]).item())
        return metrics(actual, rows)
    initial = {style: evaluate([r for r in splits["validation"] if r["format"] == style]) for style in FORMATS}
    from torch.utils.data import DataLoader
    loader = DataLoader(splits["train"], batch_size=1, shuffle=True, collate_fn=trainer.PairCollator(tok, max_tokens), num_workers=0,
                        generator=torch.Generator().manual_seed(hp.seed))
    if len(loader) % hp.grad_accum:
        raise ValueError("incomplete accumulation group")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=hp.lr, weight_decay=hp.weight_decay)
    best_loss, best_state, history = None, None, []
    for epoch in range(1, 4):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(loader):
            chosen = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rejected = trainer._reward(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
            loss = trainer.bt_loss(chosen, rejected, batch["margin"])
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            (loss / hp.grad_accum).backward()
            if (index + 1) % hp.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                save_checkpoint(root / "checkpoint")
            elapsed = time.monotonic() - started
            if elapsed > 270 * 60:
                checkpoint = root / "checkpoint"
                save_checkpoint(checkpoint)
                raise TimeoutError("CPU development training time reserve exceeded")
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        validation = trainer.evaluate(model, tok, splits["validation"], hp)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        history.append({"epoch": epoch, "validation_loss": validation["val_loss"]})
        if best_loss is None or validation["val_loss"] < best_loss:
            best_loss = validation["val_loss"]
            best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        write(root / "selection.json", {"selection_source": "validation_loss_only", "audit_used_for_selection": False, "history": history})
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(best_state[name])
    adapter = root / "adapter"
    save_checkpoint(adapter)
    validation = {style: evaluate([r for r in splits["validation"] if r["format"] == style]) for style in FORMATS}
    token_length = {}
    label_copy = {}
    for style in FORMATS:
        rows = [r for r in splits["validation"] if r["format"] == style]
        longer = sum(len(node.encode(render(r["prompt"], r["chosen"])).ids) > len(node.encode(render(r["prompt"], r["rejected"])).ids) for r in rows)
        ties = sum(len(node.encode(render(r["prompt"], r["chosen"])).ids) == len(node.encode(render(r["prompt"], r["rejected"])).ids) for r in rows)
        proportion = (longer + .5 * ties) / len(rows)
        token_length[style] = max(proportion, 1 - proportion)
        label_copy[style] = sum(1.0 if r["kind"] == "verdict" else .5 for r in rows) / len(rows)
    # These controls are observations, not candidate-selection data or an audit.
    controls = {style: {"byte_length": .5, "token_length": token_length[style], "prompt_only": .5,
                         "source_label_copy": label_copy[style], "train_fitted_answer_only": baseline["formats"][style]["accuracy"]} for style in FORMATS}
    if extra_controls:
        controls = {style: {**extra_controls[style], "train_fitted_answer_only": baseline["formats"][style]["accuracy"]} for style in FORMATS}
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import metadata, target_dims, tensor_names, write_package
    cfg = AutoConfig.from_pretrained(base, local_files_only=True)
    tensors, params = read_peft(adapter, cfg)
    training = {"family": FAMILY, "children": [{"intent": CHILD, "dataset_version": digest((root / "data/data-report.json").read_bytes()),
                "pairs": len(splits["validation"]), "accuracy": validation["json"]["accuracy"], "held_out": True}],
                "source_scope": source_report["scope"], "preference_authorship": "New AI-assisted constructed preferences from original source facts; no human review claimed.",
                "development_only": True, "independent_audit": False}
    meta = metadata(cls="judgment", intent=FAMILY, foundation_hash=sources["foundation"]["foundation_lora_fp32.onnx"],
                    foundation_version="modernbert-base-lora16-v1", rank=params["rank"], alpha=params["alpha"],
                     target_modules=params["target_modules"], max_tokens=max_tokens, training=training)
    meta.update({"x-licence": (HERE / "LICENSE").read_text(), "x-upstream-apache-2.0": (HERE / "UPSTREAM-APACHE-2.0.txt").read_text(),
                 "x-upstream-mit": (HERE / "KIT-LICENSE").read_text(), "x-source-notice": (HERE / "NOTICE").read_text()})
    package = root / (FAMILY + ".safetensors")
    write_package(package, meta, {name: tensors[name] for name in tensor_names(cfg.num_hidden_layers)})
    fixed = [node.encode(render(r["prompt"], r[side])).ids for r in splits["validation"][:8] for side in ("chosen", "rejected")]
    model.eval()
    native_fixed = []
    with torch.no_grad():
        for row in splits["validation"][:8]:
            for side in ("chosen", "rejected"):
                encoded = tok(render(row["prompt"], row[side]), truncation=False, return_tensors="pt")
                native_fixed.append(trainer._reward(model, encoded["input_ids"], encoded["attention_mask"]).item())
    model = None
    gc.collect()
    spec = {"foundation_hash": meta["foundation_hash"], "foundation_version": meta["foundation_version"], "tokenizer_hash": meta["tokenizer_hash"],
            "r_max": 16, "num_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
            "targets": {key: list(value) for key, value in target_dims(cfg).items()}}
    write(root / "foundation.json", spec)
    snapshot_sha = class_snapshot(root)
    command = [str(root / "packagecheck"), "--intent", FAMILY, "--child", CHILD, "--class-snapshot", str(root / "class-snapshot.json"),
               "--expected-max-tokens", str(max_tokens),
               "--snapshot-sha256", snapshot_sha, "--package", str(package), "--sha256", digest(package.read_bytes()),
               "--foundation", str(root / "foundation.json"), "--foundation-sha256", digest((root / "foundation.json").read_bytes()), "--out", str(root)]
    checked = subprocess.run(command, capture_output=True, text=True, timeout=60)
    write(root / "parser-result.json", {"exit": checked.returncode, "stdout": checked.stdout, "stderr": checked.stderr})
    if checked.returncode:
        raise ValueError("actual full parser refused the development package")
    manifest = json.loads(checked.stdout)
    keccak = manifest["keccak256"]
    bare = keccak[2:] if keccak.startswith("0x") else keccak
    identity = {"foundation_hash": sources["foundation"]["foundation_lora_fp32.onnx"], "engine_hash": sources["foundation"]["foundation_engine.wasm"],
                "tokenizer_hash": sources["foundation"]["tokenizer.json"], "adapter_hash": bare}
    runtime = execute(root, package, identity, fixed, max_tokens=max_tokens)
    from runtime_contract import score_value
    reference_fixed = [score_value(vector["score_bits"]) for vector in runtime["vectors"]]
    errors = [abs(a - b) for a, b in zip(native_fixed, reference_fixed)]
    if len(errors) != len(fixed) or any(error > .001 + .001 * abs(native) for error, native in zip(errors, native_fixed)):
        raise ValueError("native and real reference engine fixed vectors disagree")
    if any((native_fixed[i] > native_fixed[i + 1]) != (reference_fixed[i] > reference_fixed[i + 1]) for i in range(0, len(fixed), 2)):
        raise ValueError("native and real reference ordering disagree")
    runtime["native_reference_fixed_agreement"] = {"max_absolute_error": max(errors), "count": len(errors), "exact_bits_claimed_for_native": False}
    quality = {style: validation[style]["accuracy"] >= .8 and min(validation[style]["per_label"].values()) >= .7
               and validation[style]["accuracy"] + 1e-9 >= initial[style]["accuracy"] + .05
               and validation[style]["accuracy"] + 1e-9 >= max(controls[style].values()) + .10 for style in FORMATS}
    write(root / "development-report.json", {"schema_version": 1, "status": "DEVELOPMENT_CHECKPOINT", "family": FAMILY,
          "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"], "data": source_report,
          "dataset_gate": gate, "validation": validation, "untrained_validation": initial, "answer_only_baseline": baseline, "shortcuts": controls,
          "selection": history, "audit_used_for_selection": False, "audit_pairs": 0, "registration_ready": False,
          "package_sha256": digest(package.read_bytes()), "package_keccak": bare, "runtime": runtime, "foundation": identity,
          "development_mode": development_mode, "max_tokens": max_tokens, "development_quality_passed": all(quality.values()), "development_quality_by_format": quality,
          "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy", "safetensors")},
          "limits": ["Validation is development evidence, not an independent audit. No registration is allowed.", source_report["scope"]]})


if __name__ == "__main__":
    main()
