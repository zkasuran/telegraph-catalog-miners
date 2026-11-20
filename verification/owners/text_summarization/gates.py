# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pinned upstream source gate with complete rendering and immutable controls."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

from baseline import load_controls
from curriculum import check_tokens, warmup_rows
from data import (
    HERE,
    TOKENIZER_SHA,
    VIEWS,
    digest,
    fetch,
    prepare,
    read_splits,
    render,
    view,
    write_json,
)
from role_bootstrap import warmup_plan
from role_bootstrap import warmup_rows as role_rows
from training_policy import EPOCHS

TRAINER_SHA = "5da2919c66c2d0840104628ab88e98a9c8e5e578686f71ca9be4dcd91bd18d71"


def load_trainer(kit):
    source = kit / "float_runtime/train_adapter.py"
    if digest(source.read_bytes()) != TRAINER_SHA:
        raise ValueError("pinned reference training source mismatch")
    sys.path.insert(0, str(kit))
    spec = importlib.util.spec_from_file_location("summary_pinned_trainer", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_gate(root, kit):
    from tokenizers import Tokenizer
    prepare(root)
    report, splits = read_splits(root)
    trainer = load_trainer(kit)
    hp = trainer.Hyperparams(max_length=256, batch_size=1, grad_accum=4, lora_rank=8,
                             lora_alpha=32, epochs=EPOCHS, eval_batch=1, grad_checkpointing=False,
                             early_stopping=False, gate_min_acc=.80, lr=.0001, lora_dropout=0.0)
    gate = trainer.gate_dataset(splits["train"], hp)
    if gate["metrics"].get("surface_probe_acc") is None:
        gate["ok"] = False
        gate["failures"].append("upstream answer-only surface probe did not run")
    tokenizer_path = root / "tokenizer.json"
    fetch("https://github.com/telegraphprotocol/telegraph-foundation/releases/download/modernbert-base-lora16-v1/tokenizer.json",
          tokenizer_path, TOKENIZER_SHA, 4 << 20)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    vectors = json.loads((kit / "float_runtime/testdata/tokenizer_vectors.json").read_text())["vectors"]
    if not vectors:
        raise ValueError("missing primary tokenizer conformance vectors")
    for v in vectors:
        if tokenizer.encode(v["text"], add_special_tokens=v["add_special_tokens"]).ids != v["ids"]:
            raise ValueError("primary tokenizer conformance failed")
    checked = 0
    for rows in splits.values():
        for row in rows:
            for style in VIEWS:
                formatted = view(row, style)
                tokens = [tokenizer.encode(render(formatted["prompt"], formatted[side])).ids for side in ("chosen", "rejected")]
                if tokens[0] == tokens[1] or max(map(len, tokens)) > 256:
                    raise ValueError("whole decisive references and answers do not fit")
                checked += 2
    warmup = check_tokens(warmup_rows(json.loads((HERE / "data/manifest.json").read_text()), splits["train"]), tokenizer, render)
    bootstrap_rows = role_rows(json.loads((HERE / "data/manifest.json").read_text()), splits["train"])
    bootstrap_tokens = check_tokens(bootstrap_rows, tokenizer, render)
    controls = load_controls(splits["train"], tokenizer)
    development = {style: controls([view(r, style) for r in splits["validation"]]) for style in VIEWS}
    failures = list(gate["failures"])
    if any(v > .65 for values in development.values() for v in values.values()):
        failures.append("fixed development shortcut exceeds 65 percent")
    result = {"ok": gate["ok"] and not failures, "upstream": gate, "development_controls": development,
              "complete_rendered_inputs": checked, "tokenizer_vectors": len(vectors),
              "failures": failures, "audit_scored": False, "baseline_refit": False,
               "source_manifest_sha256": report["source_manifest_sha256"], "source_boundary": report["boundary"]}
    result.update(warmup)
    result["role_bootstrap"] = {"unique_training_pairs": len(bootstrap_rows), "complete_inputs": bootstrap_tokens["complete_warmup_inputs"],
                                "maximum_tokens": bootstrap_tokens["maximum_warmup_tokens"], "optimizer_steps": len(warmup_plan(bootstrap_rows))}
    write_json(root / "dataset-gate.json", result)
    if "torch" in sys.modules:
        raise RuntimeError("local source gate imported a neural runtime")
    return trainer, hp, splits, controls, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    args = parser.parse_args()
    *_, result = run_gate(args.root, args.kit)
    print(json.dumps(result))
    raise SystemExit(0 if result["ok"] else 1)
