# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only continuation with all-format validation and a quarantined fresh audit."""
import argparse
import gc
import http.client
import json
import math
import os
import random
from pathlib import Path

import retrain as core
from common import json_bytes
from contracts import FAMILY, FORMATS, FOUNDATION, HERE, OWNER, file_sha, finite_bits, remote_only, strict_json, validate_ready
from remote import executor, fixed_vectors, identity_controls, score, stage_check
from v3_data import BASELINE, LABELS, QUARANTINE, baselines, dump, pairs, prepare

WARM = HERE / "model/SENTIMENT_ANALYSIS-v2-failed.safetensors"


def config():
    return strict_json((HERE / "v3-config.json").read_bytes())


def source(root):
    groups, manifest = prepare(root / "data")
    cfg = config()
    for split in ("train", "validation", "audit"):
        if manifest["files"][split]["sha256"] != cfg[split + "_sha256"]:
            raise ValueError("new frozen split identity mismatch")
    for path, key in ((root / "data/source-manifest.json", "source_manifest_sha256"),
                      (BASELINE, "baseline_sha256"), (QUARANTINE, "quarantine_sha256"), (WARM, "warm_package_sha256")):
        if file_sha(path) != cfg[key]:
            raise ValueError("frozen v3 source/checkpoint/coefficient hash mismatch")
    return groups, manifest


def selection_key(metrics):
    formats = metrics["formats"]
    if set(formats) != set(FORMATS):
        raise ValueError("validation must measure every claimed format")
    values = []
    for item in formats.values():
        if set(item["per_label"]) != set(LABELS):
            raise ValueError("validation must measure every claimed label")
        values += [item["accuracy"], *item["per_label"].values()]
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("invalid actual validation metric")
    if (type(metrics.get("pairwise_accuracy")) not in (int, float) or not 0 <= metrics["pairwise_accuracy"] <= 1 or
            type(metrics.get("val_loss")) not in (int, float) or not math.isfinite(metrics["val_loss"]) or metrics["val_loss"] < 0):
        raise ValueError("invalid aggregate validation metric")
    return (min(value for item in formats.values() for value in item["per_label"].values()),
            min(item["accuracy"] for item in formats.values()), metrics["pairwise_accuracy"], -metrics["val_loss"])


def evaluate(model, tokenizer, views, hp):
    remote_only()
    import torch
    from float_runtime import train_adapter as trainer
    from torch.utils.data import DataLoader
    model.eval()
    formats = {}
    losses = []
    with torch.no_grad():
        for name, rows in views.items():
            loader = DataLoader(rows, batch_size=hp.eval_batch, shuffle=False,
                                collate_fn=trainer.PairCollator(tokenizer, hp.max_length))
            wins = []
            for batch in loader:
                chosen = model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]).logits.squeeze(-1)
                rejected = model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"]).logits.squeeze(-1)
                loss = trainer.bt_loss(chosen, rejected, batch["margin"])
                losses.append(loss.item())
                wins.extend((chosen > rejected).tolist())
            formats[name] = {"accuracy": sum(wins)/len(wins),
                             "per_label": {label: sum(win for row, win in zip(rows, wins) if row["label"] == label) /
                                           sum(row["label"] == label for row in rows) for label in LABELS}}
    result = {"formats": formats, "pairwise_accuracy": sum(item["accuracy"] for item in formats.values()) / len(formats),
              "val_loss": sum(losses)/len(losses)}
    selection_key(result)
    return result


def checkpoint(root, model, tokenizer, metrics, validation_pairs, manifest, hp):
    remote_only()
    core.save_checkpoint(root, model, tokenizer, metrics, validation_pairs, manifest, hp,
                         config_fn=config, warm_path=WARM, package_name="SENTIMENT_ANALYSIS-v3.safetensors",
                         selection_rule="worst label, worst format, mean validation accuracy then validation loss; frozen before fresh audit")


def train(root):
    remote_only()
    groups, manifest = source(root)
    base = core.base_assets(root, config_fn=config)
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
    training = pairs(groups["train"], training=True)
    validation = {name: pairs(groups["validation"], style) for style, name in enumerate(FORMATS)}
    gate = trainer.gate_dataset(training, hp)
    dump(root / "dataset-gate.json", gate)
    if not gate["ok"]:
        raise ValueError("unchanged primary dataset gate refused v3 training")
    model, tokenizer = trainer.build_model(str(base), hp)
    node_tokenizer = Tokenizer.from_file(str(root / "foundation/tokenizer.json"))
    node_tokenizer.no_padding()
    node_tokenizer.no_truncation()
    for row in training + [row for rows in validation.values() for row in rows]:
        for side in ("chosen", "rejected"):
            text = trainer.render(row["prompt"], row[side])
            tokens = node_tokenizer.encode(text).ids
            if len(tokens) > hp.max_length or tokens != tokenizer(text, truncation=False)["input_ids"]:
                raise ValueError("source claim truncated or tokenizer contract mismatch")
    core.warm_start(model, hp, package_path=WARM, expected_sha=config()["warm_package_sha256"])
    initial = evaluate(model, tokenizer, validation, hp)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=hp.lr, weight_decay=hp.weight_decay)
    loader = DataLoader(training, batch_size=hp.batch_size, shuffle=True,
                        collate_fn=trainer.PairCollator(tokenizer, hp.max_length))
    best_key = None
    best_state = None
    history = []
    for epoch in range(hp.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index, batch in enumerate(loader):
            chosen = model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]).logits.squeeze(-1)
            rejected = model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"]).logits.squeeze(-1)
            loss = trainer.bt_loss(chosen, rejected, batch["margin"])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite source training loss")
            (loss / hp.grad_accum).backward()
            losses.append(loss.item())
            if (index + 1) % hp.grad_accum == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if index % 100 == 0:
                print(json.dumps({"phase": "train", "epoch": epoch, "batch": index, "loss": loss.item()}), flush=True)
        metrics = evaluate(model, tokenizer, validation, hp)
        key = selection_key(metrics)
        history.append({"epoch": epoch, "training_loss": sum(losses)/len(losses), "validation": metrics, "selection_key": key})
        if best_key is None or key > best_key:
            best_key = key
            best_state = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
            checkpoint(root, model, tokenizer, metrics, sum(len(rows) for rows in validation.values()), manifest, hp)
        dump(root / "validation-history.json", {"initial": initial, "epochs": history})
    if best_state is None:
        raise ValueError("no validation-selected candidate")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in best_state:
                parameter.copy_(best_state[name])
    selected = evaluate(model, tokenizer, validation, hp)
    checkpoint(root, model, tokenizer, selected, sum(len(rows) for rows in validation.values()), manifest, hp)
    if (selection_key(selected)[0] < .7 or selection_key(selected)[1] < .8 or
            selected["pairwise_accuracy"] < initial["pairwise_accuracy"] + .05):
        raise ValueError("v3 failed all-format validation improvement; licensed checkpoint retained, no audit")


def audit(root):
    remote_only()
    groups, manifest = source(root)
    record = strict_json((root / "training-frozen.json").read_bytes())
    if record["run_id"] != int(os.environ["GITHUB_RUN_ID"]) or record["commit"] != os.environ["GITHUB_SHA"]:
        raise ValueError("frozen v3 training run/source identity mismatch")
    package = root / "SENTIMENT_ANALYSIS-v3.safetensors"
    if file_sha(package) != record["package_sha256"]:
        raise ValueError("frozen audit package bytes changed")
    base = core.base_assets(root, config_fn=config)
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
    views = {name: pairs(groups["audit"], style) for style, name in enumerate(FORMATS)}
    node_tokenizer = Tokenizer.from_file(str(root / "foundation/tokenizer.json"))
    node_tokenizer.no_padding()
    node_tokenizer.no_truncation()
    tokens = {name: [node_tokenizer.encode(trainer.render(row["prompt"], row[side])).ids for row in rows
                     for side in ("chosen", "rejected")] for name, rows in views.items()}
    if any(not 1 <= len(ids) <= hp.max_length for rows in tokens.values() for ids in rows):
        raise ValueError("new audit claim exceeds trained token limit")
    initial = evaluate(untrained, tokenizer, views, hp)
    del untrained
    gc.collect()
    parsed = stage_check(root, package, record["package_sha256"], "audit-parser")
    coefficient = strict_json(BASELINE.read_bytes())
    quality = {}
    actual = {}
    with executor(root, package, record["package_keccak"]) as (backend, row, server):
        negative_controls = identity_controls(server, row)
        vectors = fixed_vectors()
        first, repeated = score(backend, row, vectors), score(backend, row, vectors)
        if first != repeated:
            raise ValueError("actual fixed reference bits are not identical")
        request = {"batch": [{"tokens": ids} for ids in vectors], "foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
                   "adapter_hash": record["package_keccak"], "tokenizer_hash": FOUNDATION["tokenizer.json"]}
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=180)
        try:
            connection.request("POST", "/runnable", json_bytes(request), {"Content-Type": "application/json"})
            response = connection.getresponse()
            body = strict_json(response.read((1 << 20) + 1))
            if response.status != 200 or any(body.get(key) != request[key] for key in ("foundation_hash", "adapter_hash", "tokenizer_hash")) or body.get("results") != [{"score": bits} for bits in first]:
                raise ValueError("actual literal public wire contract failed")
        finally:
            connection.close()
        for name, rows in views.items():
            bits = score(backend, row, tokens[name])
            actual[name] = bits
            wins = [finite_bits(bits[2*i]) > finite_bits(bits[2*i+1]) for i in range(len(rows))]
            shortcut = baselines(rows, coefficient)
            quality[name] = {"accuracy": sum(wins)/len(wins),
                             "per_label": {label: sum(win for original, win in zip(rows, wins) if original["label"] == label) /
                                           sum(original["label"] == label for original in rows) for label in LABELS},
                             "untrained_accuracy": initial["formats"][name]["accuracy"],
                             "untrained_scope": "Actual seed-matched fresh PyTorch adapter on this new audit, not native scores substituted for reference scores",
                             "shortcut_accuracy": max(shortcut.values()), "shortcuts": shortcut,
                             "utf8_length_range": [min(len(group["text"].encode()) for group in groups["audit"]),
                                                   max(len(group["text"].encode()) for group in groups["audit"])]}
    passed = all(item["accuracy"] >= .8 and min(item["per_label"].values()) >= .7 and
                 item["accuracy"] >= item["untrained_accuracy"] + .05 and item["accuracy"] >= item["shortcut_accuracy"] + .1
                 for item in quality.values())
    dump(root / "actual-quality.json", {"passed": passed, "formats": quality})
    dump(root / "audit-score-bits.json", actual)
    if not passed:
        raise ValueError("new independent v3 audit failed unchanged quality gates; no ready report")
    report = {"schema_version": 1, "status": "REGISTRATION_READY", "family": FAMILY, "owner": OWNER,
              "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"], "serving_commit": os.environ["GITHUB_SHA"],
              "package_sha256": record["package_sha256"], "package_keccak": record["package_keccak"], "package_bytes": package.stat().st_size,
              "foundation_sha256": FOUNDATION["foundation_lora_fp32.onnx"], "engine_sha256": FOUNDATION["foundation_engine.wasm"],
              "tokenizer_sha256": FOUNDATION["tokenizer.json"], "supported_labels": list(LABELS),
              "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True,
                         "source_manifest_sha256": file_sha(root / "data/source-manifest.json"), "independent_groups": len(groups["audit"]),
                         "quarantine_sha256": file_sha(QUARANTINE), "construction": manifest["construction"]},
              "quality": {"passed": passed, "audit_pairs": len(views["json"]), "all_supported_formats_tested": True, "formats": quality},
              "runtime": {"full_parser_passed": parsed["manifest"]["verification"]["stage1"], "reference_engine_passed": True,
                          "repeat_bits_identical": True, "negative_identity_controls": negative_controls,
                          "vectors": [{"tokens": ids, "score_bits": bits} for ids, bits in zip(vectors, first)]},
              "training": record, "limits": [manifest["scope"], "A source-held-out result is not canonical qualification."]}
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
        dump(args.root / "failure.json", {"status": "FAILED", "family": FAMILY, "phase": args.phase,
                                         "run_id": os.environ["GITHUB_RUN_ID"], "commit": os.environ["GITHUB_SHA"],
                                         "error_type": type(error).__name__, "message": str(error)[:512]})
        raise
