# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only warm-start training, frozen validation selection and independent audit."""
import argparse
import gc
import http.client
import importlib.metadata
import json
import math
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

from contracts import FAMILY, FORMATS, FOUNDATION, HERE, OWNER, file_sha, finite_bits, package_parts, remote_only, strict_json, validate_ready
from common import json_bytes
from remote import assets, executor, fixed_vectors, identity_controls, score, stage_check
from retrain_data import LABELS, baselines, dump, make_pairs, prepare


def config():
    return strict_json((HERE / "retrain-config.json").read_bytes())


def source(root):
    groups, manifest = prepare(root / "data")
    cfg = config()
    for name in ("train", "validation", "audit"):
        if manifest["files"][name]["sha256"] != cfg[name + "_sha256"]:
            raise ValueError("frozen new split identity mismatch")
    if (file_sha(root / "data/source-manifest.json") != cfg["source_manifest_sha256"] or
            file_sha(HERE / "retrain-baseline.json") != cfg["baseline_sha256"] or
            file_sha(HERE / "retrain-sources.json") != cfg["sources_sha256"]):
        raise ValueError("frozen source or coefficient bytes changed")
    return groups, manifest


def base_assets(root, config_fn=config):
    remote_only()
    cfg = config_fn()
    assets(root)
    base = root / "base"
    base.mkdir(exist_ok=True)
    for name, expected in cfg["base_files"].items():
        path = base / name
        if path.is_file() and file_sha(path) == expected:
            continue
        url = "https://huggingface.co/answerdotai/ModernBERT-base/resolve/" + cfg["base_revision"] + "/" + name
        request = urllib.request.Request(url, headers={"User-Agent": "Sentiment-owner-training/2.0"})
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=30) as response, path.open("wb") as out:
            total = 0
            while chunk := response.read(1 << 20):
                total += len(chunk)
                if total > 1 << 30 or time.monotonic() - started > 600:
                    raise ValueError("base source byte ceiling")
                out.write(chunk)
        if file_sha(path) != expected:
            raise ValueError("pinned base source hash mismatch")
    (base / "tokenizer.json").write_bytes((root / "foundation/tokenizer.json").read_bytes())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(root / ("telegraph-wasm-baseline-" + strict_json((HERE / "frozen.json").read_bytes())["kit_commit"])))
    return base


def warm_start(model, hp, package_path=None, expected_sha=None):
    remote_only()
    import torch
    from safetensors.torch import load_file
    package = package_path or HERE / "model/SENTIMENT_ANALYSIS-recovery.safetensors"
    if file_sha(package) != (expected_sha or config()["warm_package_sha256"]):
        raise ValueError("warm-start package identity mismatch")
    metadata = package_parts(package.read_bytes())[0]["__metadata__"]
    if int(metadata["rank"]) != hp.lora_rank or float(metadata["alpha"]) != hp.lora_alpha:
        raise ValueError("warm-start rank or scale mismatch")
    tensors = load_file(str(package))
    with torch.no_grad():
        for layer in range(22):
            for target in ("attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"):
                module = model.get_submodule(f"base_model.model.model.layers.{layer}.{target}")
                module.lora_A["default"].weight.copy_(tensors[f"lora.{layer}.{target}.A"].T)
                module.lora_B["default"].weight.copy_(tensors[f"lora.{layer}.{target}.B"].T)
        head = model.get_submodule("base_model.model.classifier").modules_to_save["default"]
        head.weight.copy_(tensors["head.weight"].T)
        if head.bias is not None:
            head.bias.copy_(tensors["head.bias"])
        elif tensors["head.bias"].count_nonzero().item() != 0:
            raise ValueError("nonzero package bias cannot be represented by the pinned bias-free head")


def save_checkpoint(root, model, tokenizer, selected, validation_pairs, manifest, hp,
                    config_fn=config, warm_path=None, package_name="SENTIMENT_ANALYSIS-v2.safetensors",
                    selection_rule="validation loss only; frozen before new independent audit"):
    remote_only()
    from float_runtime import train_adapter as trainer
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import metadata, tensor_names, write_package
    from transformers import AutoConfig
    adapter = root / "adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    trainer.save_reward_head(model, adapter)
    tensors, parameters = read_peft(adapter, AutoConfig.from_pretrained(root / "base", local_files_only=True))
    cfg = config_fn()
    original = warm_path or HERE / "model/SENTIMENT_ANALYSIS-recovery.safetensors"
    if file_sha(original) != cfg["warm_package_sha256"]:
        raise ValueError("checkpoint warm-start lineage mismatch")
    lineage = package_parts(original.read_bytes())[0]["__metadata__"]
    record = {"family": FAMILY, "fitted_material": True, "source_manifest_sha256": file_sha(root / "data/source-manifest.json"),
              "children": [{"intent": FAMILY + "_GENERAL", "dataset_version": cfg["validation_sha256"],
                            "pairs": validation_pairs, "accuracy": selected["pairwise_accuracy"], "held_out": False}],
              "selection": selection_rule, "supported_labels": list(LABELS),
              "warm_start_sha256": cfg["warm_package_sha256"], "run_id": int(os.environ["GITHUB_RUN_ID"]),
              "commit": os.environ["GITHUB_SHA"], "preference_authorship": manifest["construction"]}
    meta = metadata(cls="judgment", intent=FAMILY, foundation_hash=FOUNDATION["foundation_lora_fp32.onnx"],
                    foundation_version="modernbert-base-lora16-v1", rank=parameters["rank"], alpha=parameters["alpha"],
                    target_modules=parameters["target_modules"], max_tokens=hp.max_length, training=record)
    for key in ("x-licence", "x-upstream-notice", "x-upstream-apache-2.0", "x-upstream-mit"):
        meta[key] = lineage[key]
    meta["x-goemotions-apache-2.0"] = (root / "data/LICENSE").read_text()
    meta["x-goemotions-attribution"] = "GoEmotions, Demszky et al. (2020), Google Research. Emotion annotations retained; valence preferences and Mixed juxtapositions constructed by AI-assisted code."
    if package_name not in ("SENTIMENT_ANALYSIS-v2.safetensors", "SENTIMENT_ANALYSIS-v3.safetensors", "SENTIMENT_ANALYSIS-v4.safetensors"):
        raise ValueError("unsupported owned checkpoint name")
    package = root / package_name
    write_package(package, meta, {name: tensors[name] for name in tensor_names(22)})
    if package_parts(package.read_bytes())[1] == package_parts(original.read_bytes())[1]:
        raise ValueError("unchanged tensor retry refused")
    parsed = stage_check(root, package, file_sha(package), "new-parser")
    record.update(package_sha256=file_sha(package), package_keccak=parsed["keccak256"].removeprefix("0x"),
                  validation=selected, hyperparameters=cfg["hyperparameters"],
                  versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "safetensors")})
    dump(root / "training-frozen.json", record)


def train(root):
    remote_only()
    groups, manifest = source(root)
    base = base_assets(root)
    import numpy as np
    import torch
    from float_runtime import train_adapter as trainer
    from tokenizers import Tokenizer
    from torch.utils.data import DataLoader
    hp = trainer.Hyperparams(**config()["hyperparameters"])
    torch.set_num_threads(4)
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    training = make_pairs(groups["train"], training=True)
    validation = make_pairs(groups["validation"])
    gate = trainer.gate_dataset(training, hp)
    dump(root / "dataset-gate.json", gate)
    if not gate["ok"]:
        raise ValueError("unchanged primary dataset gate refused new training pairs")
    model, tokenizer = trainer.build_model(str(base), hp)
    node_tokenizer = Tokenizer.from_file(str(root / "foundation/tokenizer.json"))
    node_tokenizer.no_padding()
    node_tokenizer.no_truncation()
    for row in training + validation:
        for side in ("chosen", "rejected"):
            text = trainer.render(row["prompt"], row[side])
            expected = node_tokenizer.encode(text).ids
            if len(expected) > hp.max_length or expected != tokenizer(text, truncation=False)["input_ids"]:
                raise ValueError("source claim truncated or tokenizer mismatch")
    warm_start(model, hp)
    initial = trainer.evaluate(model, tokenizer, validation, hp)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=hp.lr, weight_decay=hp.weight_decay)
    loader = DataLoader(training, batch_size=hp.batch_size, shuffle=True,
                        collate_fn=trainer.PairCollator(tokenizer, hp.max_length))
    history = []
    best = math.inf
    best_state = None
    for epoch in range(hp.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index, batch in enumerate(loader):
            chosen = model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]).logits.squeeze(-1)
            rejected = model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"]).logits.squeeze(-1)
            loss = trainer.bt_loss(chosen, rejected, batch["margin"])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            (loss / hp.grad_accum).backward()
            losses.append(loss.item())
            if (index + 1) % hp.grad_accum == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if index % 100 == 0:
                print(json.dumps({"phase": "train", "epoch": epoch, "batch": index, "loss": loss.item()}), flush=True)
        metrics = trainer.evaluate(model, tokenizer, validation, hp)
        history.append({"epoch": epoch, "training_loss": sum(losses)/len(losses), "validation": metrics})
        if metrics["val_loss"] < best:
            best = metrics["val_loss"]
            best_state = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
            save_checkpoint(root, model, tokenizer, metrics, len(validation), manifest, hp)
        dump(root / "validation-history.json", {"initial": initial, "epochs": history})
    if best_state is None:
        raise ValueError("no frozen validation candidate")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in best_state:
                parameter.copy_(best_state[name])
    selected = trainer.evaluate(model, tokenizer, validation, hp)
    save_checkpoint(root, model, tokenizer, selected, len(validation), manifest, hp)
    if selected["pairwise_accuracy"] < .8 or selected["pairwise_accuracy"] < initial["pairwise_accuracy"] + .05:
        raise ValueError("new candidate failed validation improvement gate; licensed checkpoint retained, no audit performed")


def audit(root):
    remote_only()
    groups, manifest = source(root)
    record = strict_json((root / "training-frozen.json").read_bytes())
    if record["commit"] != os.environ["GITHUB_SHA"] or record["run_id"] != int(os.environ["GITHUB_RUN_ID"]):
        raise ValueError("frozen training run/source mismatch")
    package = root / "SENTIMENT_ANALYSIS-v2.safetensors"
    if file_sha(package) != record["package_sha256"]:
        raise ValueError("frozen candidate package changed")
    base = base_assets(root)
    import numpy as np
    import torch
    from float_runtime import train_adapter as trainer
    from tokenizers import Tokenizer
    hp = trainer.Hyperparams(**config()["hyperparameters"])
    torch.set_num_threads(4)
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    untrained, tokenizer = trainer.build_model(str(base), hp)
    views = {name: make_pairs(groups["audit"], style) for style, name in enumerate(FORMATS)}
    initial = {name: trainer.evaluate(untrained, tokenizer, rows, hp)["pairwise_accuracy"] for name, rows in views.items()}
    del untrained
    gc.collect()
    node_tokenizer = Tokenizer.from_file(str(root / "foundation/tokenizer.json"))
    node_tokenizer.no_padding()
    node_tokenizer.no_truncation()
    token_views = {}
    for name, rows in views.items():
        token_views[name] = [node_tokenizer.encode(trainer.render(row["prompt"], row[side])).ids
                             for row in rows for side in ("chosen", "rejected")]
        if any(not 1 <= len(ids) <= hp.max_length for ids in token_views[name]):
            raise ValueError("new independent audit claim exceeds trained token bound")
    parsed = stage_check(root, package, record["package_sha256"], "audit-parser")
    coefficient = strict_json((HERE / "retrain-baseline.json").read_bytes())
    quality = {}
    actual_bits = {}
    with executor(root, package, record["package_keccak"]) as (backend, row, server):
        negatives = identity_controls(server, row)
        vectors = fixed_vectors()
        first, second = score(backend, row, vectors), score(backend, row, vectors)
        if first != second:
            raise ValueError("actual reference repeat bits differ")
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=180)
        try:
            request = {"batch": [{"tokens": ids} for ids in vectors],
                       "foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
                       "adapter_hash": record["package_keccak"], "tokenizer_hash": FOUNDATION["tokenizer.json"]}
            connection.request("POST", "/runnable", json_bytes(request), {"Content-Type": "application/json"})
            response = connection.getresponse()
            body = strict_json(response.read((1 << 20) + 1))
            if (response.status != 200 or any(body.get(key) != request[key] for key in
                    ("foundation_hash", "adapter_hash", "tokenizer_hash")) or
                    body.get("results") != [{"score": bit} for bit in first]):
                raise ValueError("actual public front-end literal wire contract failed")
        finally:
            connection.close()
        for name, rows in views.items():
            bits = score(backend, row, token_views[name])
            actual_bits[name] = bits
            wins = [finite_bits(bits[2*i]) > finite_bits(bits[2*i+1]) for i in range(len(rows))]
            shortcuts = baselines(rows, coefficient)
            quality[name] = {"accuracy": sum(wins)/len(wins),
                             "per_label": {label: sum(w for r, w in zip(rows, wins) if r["label"] == label) /
                                           sum(r["label"] == label for r in rows) for label in LABELS},
                             "untrained_accuracy": initial[name], "untrained_scope": "seed-matched fresh PyTorch adapter, actual new audit",
                             "shortcut_accuracy": max(shortcuts.values()), "shortcuts": shortcuts,
                             "utf8_length_range": [min(len(r["text"].encode()) for r in groups["audit"]),
                                                   max(len(r["text"].encode()) for r in groups["audit"])]}
    passed = all(m["accuracy"] >= .8 and min(m["per_label"].values()) >= .7 and
                 m["accuracy"] >= m["untrained_accuracy"] + .05 and m["accuracy"] >= m["shortcut_accuracy"] + .1
                 for m in quality.values())
    dump(root / "actual-quality.json", {"passed": passed, "formats": quality})
    dump(root / "audit-score-bits.json", actual_bits)
    if not passed:
        raise ValueError("independent audit failed unchanged quality gates; never emit ready")
    report = {"schema_version": 1, "status": "REGISTRATION_READY", "family": FAMILY, "owner": OWNER,
              "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"], "serving_commit": os.environ["GITHUB_SHA"],
              "package_sha256": record["package_sha256"], "package_keccak": record["package_keccak"], "package_bytes": package.stat().st_size,
              "foundation_sha256": FOUNDATION["foundation_lora_fp32.onnx"], "engine_sha256": FOUNDATION["foundation_engine.wasm"],
              "tokenizer_sha256": FOUNDATION["tokenizer.json"], "supported_labels": list(LABELS),
              "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True,
                         "source_manifest_sha256": file_sha(root / "data/source-manifest.json"), "independent_groups": len(groups["audit"]),
                         "boundary": "new GoEmotions author/thread/time-held-out groups; no original UCI audit or exposed node pairs",
                         "construction": manifest["construction"]},
              "quality": {"passed": passed, "audit_pairs": len(views["json"]), "all_supported_formats_tested": True, "formats": quality},
              "runtime": {"full_parser_passed": parsed["manifest"]["verification"]["stage1"], "reference_engine_passed": True,
                          "repeat_bits_identical": True, "negative_identity_controls": negatives,
                          "vectors": [{"tokens": ids, "score_bits": bit} for ids, bit in zip(vectors, first)]},
              "training": record,
              "limits": [manifest["scope"], "Reason correctness and numeric confidence are not independently validated."]}
    validate_ready(report)
    dump(root / "registration-ready.json", report)


if __name__ == "__main__":
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("train", "audit"))
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    try:
        {"train": train, "audit": audit}[args.phase](args.root)
    except Exception as error:
        dump(args.root / "failure.json", {"status": "FAILED", "phase": args.phase, "family": FAMILY,
                                         "run_id": os.environ["GITHUB_RUN_ID"], "commit": os.environ["GITHUB_SHA"],
                                         "error_type": type(error).__name__, "message": str(error)[:512]})
        raise
