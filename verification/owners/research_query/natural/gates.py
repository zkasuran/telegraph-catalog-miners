# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Unchanged source-only quality gates and exact train-fitted baseline replay."""
import importlib.util
import json
import math
import re
import sys
from collections import Counter

from data import HERE, digest, read_splits, render, write_json

WORDS = re.compile(r"[a-z0-9']+")


def formatted(rows, name):
    return [dict(row) for row in rows] if name == "prose" else [{**row, **{side: json.dumps({"answer": row[side]}, ensure_ascii=False, separators=(",", ":")) for side in ("chosen", "rejected")}} for row in rows]


def controls(rows, frozen, tokenizer):
    values = {name: [] for name in ("length", "token_length", "overlap", "answer_only", "citation", "hedging")}
    for row in rows:
        c, r = Counter(WORDS.findall(row["chosen"].casefold())), Counter(WORDS.findall(row["rejected"].casefold()))
        p = set(WORDS.findall(row["prompt"].casefold()))
        def bow(vector):
            return math.fsum(vector[word] * weight for word, weight in frozen["coefficients"].items()) / max(math.sqrt(sum(n * n for n in vector.values())), 1)
        values["length"].append(len(row["chosen"]) - len(row["rejected"]))
        values["token_length"].append(len(tokenizer.encode(render(row["prompt"], row["chosen"])).ids) - len(tokenizer.encode(render(row["prompt"], row["rejected"])).ids))
        values["overlap"].append(sum(c[t] for t in p) / max(sum(c.values()), 1) - sum(r[t] for t in p) / max(sum(r.values()), 1))
        values["answer_only"].append(bow(c) - bow(r))
        values["citation"].append(row["chosen"].count("http") - row["rejected"].count("http"))
        values["hedging"].append(sum(c[w] for w in ("may", "might", "perhaps")) - sum(r[w] for w in ("may", "might", "perhaps")))
    return {name: sum(1 if v * frozen["orientations"][name] > 0 else .5 if v == 0 else 0 for v in diffs) / len(diffs) for name, diffs in values.items()}


def run_gate(directory, kit):
    from tokenizers import Tokenizer
    source = kit / "float_runtime/train_adapter.py"
    if digest(source.read_bytes()) != "5da2919c66c2d0840104628ab88e98a9c8e5e578686f71ca9be4dcd91bd18d71":
        raise ValueError("actual primary trainer code changed")
    spec = importlib.util.spec_from_file_location("natural_pinned_trainer", source)
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    report, raw = read_splits(directory)
    contract = json.loads((HERE / "contract.json").read_text())
    frozen_raw = (directory / "frozen-controls.json").read_bytes()
    if digest(frozen_raw) != contract["frozen_controls_sha256"]:
        raise ValueError("immutable train-only coefficients changed")
    frozen = json.loads(frozen_raw)
    if frozen["fit_split"] != "train" or frozen["training_sha256"] != report["files"]["train"]["sha256"]:
        raise ValueError("baseline is not bound to original training")
    tokenizer = Tokenizer.from_file(str(directory / "sources/tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    split = {**raw, "train": [r for row in raw["train"] for r in formatted([row], "prose" if int(digest(row["group_id"].encode()), 16) % 2 == 0 else "json")],
             "validation": formatted(raw["validation"], "prose") + formatted(raw["validation"], "json")}
    hp = trainer.Hyperparams(max_length=512, batch_size=1, grad_accum=4, epochs=8, lr=.0001, lora_dropout=.05,
                            lora_rank=8, lora_alpha=32, eval_batch=1, early_stopping=False, grad_checkpointing=False, gate_min_acc=.8)
    gate = trainer.gate_dataset(split["train"], hp)
    if gate["metrics"].get("surface_probe_acc") is None:
        raise ValueError("actual source surface probe did not run")
    for name in ("prose", "json"):
        for rows in raw.values():
            for row in formatted(rows, name):
                if any(len(tokenizer.encode(render(row["prompt"], row[side])).ids) > 512 for side in ("chosen", "rejected")):
                    raise ValueError("whole original labelled input was cropped")
    validation = controls(split["validation"], frozen, tokenizer)
    if max(validation.values()) > .65:
        gate["ok"] = False
        gate["failures"].append("frozen validation shortcut exceeds unchanged 65 percent gate")
    write_json(directory / "dataset-gate.json", {**gate, "validation_controls": validation, "audit_scored": False,
               "immutable_coefficients": contract["frozen_controls_sha256"]})
    report["training_allowed"] = gate["ok"]
    write_json(directory / "data-report.json", report)
    return trainer, hp, split, lambda rows: controls(rows, frozen, tokenizer), gate
