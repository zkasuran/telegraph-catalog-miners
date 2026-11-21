# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Train and freeze one candidate on public CPU. This does not read audit rows."""
import gc
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

from assets import CHILD, FAMILY, HERE, SCOPE, capture_classes, prepare, remote_only
from common import FOUNDATION, KIT, sha, strict_json, write_json
from parent_scope import claimed_intents, training_record


def read_split(name, manifest):
    path = HERE / "frozen" / (name + ".jsonl")
    raw = path.read_bytes()
    if sha(raw) != manifest["files"][name]["sha256"]:
        raise ValueError("frozen split digest mismatch")
    # The actual pinned PairCollator indexes p["margin"], unlike the JSONL
    # reader that adds the default. Direct callers must supply it explicitly.
    return [{**strict_json(line), "margin": 0.0} for line in raw.splitlines()]


def main():
    remote_only()
    root = Path(os.environ["RUNNER_TEMP"]) / "event-outcome-training"
    root.mkdir(exist_ok=True)
    checkpoint = root / "adapter"
    checkpoint.mkdir(exist_ok=True)
    for name in ("LICENSE", "NOTICE", "DATA-SOURCES.md", "UPSTREAM-LICENCES.json"):
        (checkpoint / name).write_bytes((HERE / name).read_bytes())
    # Fail closed on live class/routing drift before reading data or fetching
    # neural assets. Keep the final capture too, so long training cannot use a
    # stale preflight as registration evidence.
    capture_classes(root / "class-preflight.json")
    frozen = strict_json((HERE / "frozen/frozen.json").read_bytes())
    train, validation = read_split("train", frozen), read_split("validation", frozen)
    foundation, base, kit = prepare(root, training=True)
    sys.path.insert(0, str(kit))
    import numpy as np
    import torch
    from float_runtime import train_adapter as trainer
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import metadata, tensor_names, write_package
    from tokenizers import Tokenizer
    from transformers import AutoConfig, AutoTokenizer

    torch.set_num_threads(4)
    hp = trainer.Hyperparams(max_length=256, batch_size=2, grad_accum=8, epochs=3,
                            eval_batch=4, gate_min_acc=0.80, grad_checkpointing=False)
    gate = trainer.gate_dataset(train, hp)
    write_json(root / "dataset-gate.json", gate)
    if not gate["ok"]:
        raise ValueError("unchanged upstream dataset gate refused training")
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    node_tok = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    node_tok.no_padding()
    node_tok.enable_truncation(max_length=256)
    checked = 0
    for row in train + validation:
        for side in ("chosen", "rejected"):
            text = trainer.render(row["prompt"], row[side])
            if len(tok(text, truncation=False)["input_ids"]) > 256 or tok(text, truncation=True, max_length=256)["input_ids"] != node_tok.encode(text).ids:
                raise ValueError("training/node tokens differ or discriminating answer truncates")
            checked += 1
    trainer.check_tokenizer(tok)
    write_json(root / "tokenizer-check.json", {"checked_texts": checked, "tokenizer_sha256": FOUNDATION["tokenizer.json"]})
    licences = strict_json((HERE / "UPSTREAM-LICENCES.json").read_bytes())
    cfg = AutoConfig.from_pretrained(base, local_files_only=True)

    def export(directory, package_name, validation_accuracy):
        tensors, params = read_peft(directory, cfg)
        if SCOPE == "parent_only":
            record = training_record(len(validation), validation_accuracy)
            record["children"][0]["dataset_version"] = sha((HERE / "frozen/frozen.json").read_bytes())
        else:
            record = {"family": FAMILY, "children": [{"intent": CHILD,
                      "dataset_version": sha((HERE / "frozen/frozen.json").read_bytes()),
                      "pairs": len(validation), "accuracy": validation_accuracy, "held_out": True}]}
        meta = metadata(cls="judgment", intent=FAMILY, foundation_hash=FOUNDATION["foundation_lora_fp32.onnx"],
                        foundation_version="modernbert-base-lora16-v1", rank=params["rank"], alpha=params["alpha"],
                        target_modules=params["target_modules"], max_tokens=256, training=record)
        meta.update({"x-licence": (HERE / "LICENSE").read_text(), "x-notice": (HERE / "NOTICE").read_text(),
                     "x-upstream-mit": licences["kit_mit"], "x-upstream-apache-2.0": licences["apache_2_0"],
                     "x-source-terms": (HERE / "DATA-SOURCES.md").read_text()})
        path = root / package_name
        write_package(path, meta, {name: tensors[name] for name in tensor_names(cfg.num_hidden_layers)})
        return {"file": package_name, "sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}

    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    initial, _ = trainer.build_model(str(base), hp)
    initdir = root / "untrained-adapter"
    initial.save_pretrained(initdir)
    trainer.save_reward_head(initial, initdir)
    initial_validation = trainer.evaluate(initial, tok, validation, hp)
    baseline_package = export(initdir, "untrained.safetensors", initial_validation["pairwise_accuracy"])
    del initial
    gc.collect()
    model, tok, _ = trainer.train(str(base), train, validation, checkpoint, hp)
    selected_validation = trainer.evaluate(model, tok, validation, hp)
    trained_package = export(checkpoint, FAMILY + ".safetensors", selected_validation["pairwise_accuracy"])
    binding = {"schema_version": 1, "family": FAMILY, "run_id": int(os.environ["GITHUB_RUN_ID"]),
               "scope": SCOPE, "claimed_intents": claimed_intents(SCOPE),
               "commit": os.environ["GITHUB_SHA"], "owner_attempt": os.environ["OWNER_ATTEMPT"],
               "package": trained_package, "untrained_package": baseline_package,
               "source_manifest_sha256": sha((HERE / "frozen/frozen.json").read_bytes()),
               "hyperparameters": asdict(hp), "selected_validation": selected_validation,
               "untrained_validation": initial_validation, "kit_commit": KIT, "foundation": FOUNDATION,
               "audit_read": False, "selection": "unchanged upstream best validation epoch",
               "audit_policy": "one subsequent frozen-package reference-engine audit; never select on audit"}
    capture_classes(root / "class-snapshot.json")
    write_json(root / "candidate.json", binding)
    print(json.dumps({"status": "CANDIDATE_FROZEN", "package_sha256": trained_package["sha256"], "run_id": binding["run_id"]}))


if __name__ == "__main__":
    main()
