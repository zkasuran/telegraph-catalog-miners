# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Train-only source gate and immutable train-fitted shortcut replay."""
import json

from data import HERE, digest, read_splits
from fit_core import fit_controls, load_trainer
from frozen_baseline import decode_frozen, replay_controls, validate_frozen


def fit_and_gate_development(train, validation, tokenizer, kit):
    if len(train) != 768 or len(validation) != 128:
        raise ValueError("generation-two development cardinality mismatch")
    trainer = load_trainer(kit)
    hp = trainer.Hyperparams(max_length=256, epochs=3)
    upstream = trainer.gate_dataset(train, hp)
    bundle = HERE / "baselines/frozen-controls.json"
    if bundle.exists():
        contract = json.loads((HERE / "contract.json").read_text())
        raw = bundle.read_bytes()
        if digest(raw) != contract["frozen_controls_sha256"]:
            raise ValueError("bundled train-only baseline changed")
        frozen = decode_frozen(raw)
    else:
        # Source-only first fit on training. This occurs before audit creation.
        _, frozen = fit_controls(train, tokenizer)
    validate_frozen(frozen, train)
    scores = replay_controls(frozen, tokenizer)
    metrics = scores(validation)
    failures = [name + " development shortcut exceeds 65 percent" for name, value in metrics.items() if value > .65]
    return {"ok": upstream["ok"] and not failures, "upstream": upstream, "validation_controls": metrics, "failures": failures,
            "train_rows": 768, "development_rows": 128, "fit_split": "train", "audit_used": False}, frozen


def run_gate(root, kit):
    from tokenizers import Tokenizer
    trainer = load_trainer(kit)
    report, splits = read_splits(root)
    contract = json.loads((HERE / "contract.json").read_text())
    if any(report["files"][name]["sha256"] != pin["sha256"] or len(splits[name]) != pin["rows"]
           for name, pin in contract["splits"].items()):
        raise ValueError("frozen generation-two split identity mismatch")
    hp = trainer.Hyperparams(max_length=256, batch_size=1, grad_accum=16, lora_rank=8, lora_alpha=32,
                            epochs=3, eval_batch=1, grad_checkpointing=False, early_stopping=False, gate_min_acc=.8)
    upstream = trainer.gate_dataset(splits["train"], hp)
    tokenizer = Tokenizer.from_file(str(root / "sources/tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    vectors = json.loads((kit / "float_runtime/testdata/tokenizer_vectors.json").read_text())["vectors"]
    if len(vectors) < 8 or any(tokenizer.encode(v["text"], add_special_tokens=v["add_special_tokens"]).ids != v["ids"] for v in vectors):
        raise ValueError("released tokenizer conformance failed")
    for rows in splits.values():
        for row in rows:
            tokens = [tokenizer.encode(trainer.render(row["prompt"], row[side])).ids for side in ("chosen", "rejected")]
            if max(map(len, tokens)) > 256 or tokens[0] == tokens[1]:
                raise ValueError("source preference is clipped or token-identical")
    development, frozen = fit_and_gate_development(splits["train"], splits["validation"], tokenizer, kit)
    result = {"ok": upstream["ok"] and development["ok"], "upstream": upstream,
              "validation_controls": development["validation_controls"], "baseline_refit": False,
              "conformance_vectors": len(vectors), "intact_pairs": sum(map(len, splits.values())),
              "audit_metrics": "not scored by data gate", "no_gate_bypass": True}
    if upstream["metrics"].get("surface_probe_acc") is None:
        result["ok"] = False
    from source_data import write_json
    write_json(root / "dataset-gate.json", result)
    return trainer, hp, splits, replay_controls(frozen, tokenizer), result
