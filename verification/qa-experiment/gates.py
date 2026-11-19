# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Source-only reference gates and train-fitted frozen shortcut controls."""
import argparse
import importlib.util
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

from data import HERE, digest, read_splits, render, write_json


def load_trainer(kit):
    source = kit / "float_runtime/train_adapter.py"
    pins = json.loads((HERE / "sources.json").read_text())
    if digest(source.read_bytes()) != pins["trainer_sha256"]:
        raise ValueError("reference trainer pin mismatch")
    sys.path.insert(0, str(kit))
    spec = importlib.util.spec_from_file_location("qa_pinned_trainer", source)
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    return trainer


WORDS = re.compile(r"[a-z0-9']+")


def words(text):
    return WORDS.findall(text.casefold())


def fixed_features(prompt, answer):
    terms = words(answer)
    lower = answer.casefold()
    return [len(answer), len(terms), answer.count("\n"), answer.count("```"),
            int("http" in lower), sum(lower.count(x) for x in ("sorry", "cannot", "can't", "unable")),
            sum(c.isdigit() for c in answer), sum(c in "!?;:" for c in answer)]


def pair_accuracy(values):
    return sum(1 if x > 0 else 0.5 if x == 0 else 0 for x in values) / len(values)


def fit_controls(train, tokenizer):
    import numpy as np
    df = Counter()
    for row in train:
        for field in ("prompt", "chosen", "rejected"):
            df.update(set(words(row[field])))
    vocab = {term: i for i, (term, _) in enumerate(sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))[:2000])}
    idf = np.array([math.log((1 + 3 * len(train)) / (1 + df[term])) + 1 for term in vocab])

    def vector(text):
        vec = np.zeros(len(vocab))
        for term in words(text):
            if term in vocab:
                vec[vocab[term]] += 1
        vec *= idf
        return vec / max(np.linalg.norm(vec), 1e-8)

    features = np.array([fixed_features(r["prompt"], r[s]) for r in train for s in ("chosen", "rejected")], dtype=float)
    scale = np.maximum(features.std(axis=0), 1)
    x = np.array([vector(r[s]) for r in train for s in ("chosen", "rejected")])
    y = np.tile([1., 0.], len(train))

    def logistic(matrix):
        weight = np.zeros(matrix.shape[1])
        for _ in range(60):
            prediction = 1 / (1 + np.exp(-np.clip(matrix @ weight, -30, 30)))
            weight -= 0.2 * (matrix.T @ (prediction - y) / len(y) + 1e-4 * weight)
        return weight

    bow, style = logistic(x), logistic(features / scale)
    # Every orientation is fit on training only and stays fixed thereafter.
    orientations = {}
    for name, diffs in {
        "length": [len(r["chosen"]) - len(r["rejected"]) for r in train],
        "token_length": [len(tokenizer.encode(render(r["prompt"], r["chosen"])).ids) - len(tokenizer.encode(render(r["prompt"], r["rejected"])).ids) for r in train],
        "overlap": [float(vector(r["prompt"]) @ (vector(r["chosen"]) - vector(r["rejected"]))) for r in train],
    }.items():
        orientations[name] = 1 if pair_accuracy(diffs) >= 0.5 else -1

    def scores(rows):
        result = {name: [] for name in ("length", "token_length", "overlap", "answer_only", "surface")}
        for row in rows:
            chosen, rejected = vector(row["chosen"]), vector(row["rejected"])
            result["length"].append(orientations["length"] * (len(row["chosen"]) - len(row["rejected"])))
            result["token_length"].append(orientations["token_length"] * (len(tokenizer.encode(render(row["prompt"], row["chosen"])).ids) - len(tokenizer.encode(render(row["prompt"], row["rejected"])).ids)))
            result["overlap"].append(orientations["overlap"] * float(vector(row["prompt"]) @ (chosen - rejected)))
            result["answer_only"].append(float((chosen - rejected) @ bow))
            difference = np.array(fixed_features(row["prompt"], row["chosen"])) - fixed_features(row["prompt"], row["rejected"])
            result["surface"].append(float((difference / scale) @ style))
        return {name: pair_accuracy(values) for name, values in result.items()}

    frozen = {"vocabulary": vocab, "idf": idf.tolist(), "bow": bow.tolist(), "style": style.tolist(),
              "scale": scale.tolist(), "orientations": orientations, "fit_split": "train"}
    return scores, frozen


def visible_controls(splits, tokenizer, trainer):
    rows_checked = 0
    for rows in splits.values():
        for row in rows:
            ids = [tokenizer.encode(trainer.render(row["prompt"], row[side])).ids for side in ("chosen", "rejected")]
            if trainer.render(row["prompt"], row["chosen"]) != render(row["prompt"], row["chosen"]):
                raise ValueError("renderer mismatch")
            if max(map(len, ids)) > 256 or ids[0] == ids[1]:
                raise ValueError("complete preference not visible under cap")
            rows_checked += 1
    return {"pairs": rows_checked, "full_inputs_fit": True, "identical_retained_pairs": 0, "max_tokens": 256}


def run_gate(root, kit):
    from tokenizers import Tokenizer
    trainer = load_trainer(kit)
    source_report, splits = read_splits(root)
    contract = json.loads((HERE / "contract.json").read_text())
    fixed_binding = source_report["training_allowed"] and all(
        source_report["files"][name]["rows"] == value["rows"] and source_report["files"][name]["sha256"] == value["sha256"]
        for name, value in contract["splits"].items())
    hp = trainer.Hyperparams(max_length=256, batch_size=1, grad_accum=16, lora_rank=8,
                            lora_alpha=32, epochs=3, eval_batch=1, grad_checkpointing=False, early_stopping=False,
                            gate_min_acc=0.80)
    gate = trainer.gate_dataset(splits["train"], hp)
    if gate["metrics"].get("surface_probe_acc") is None:
        gate["ok"] = False
        gate["failures"].append("upstream surface probe did not run")
    tokenizer = Tokenizer.from_file(str(root / "sources/tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    visible = visible_controls(splits, tokenizer, trainer)
    vector_path = kit / "float_runtime/testdata/tokenizer_vectors.json"
    vectors = json.loads(vector_path.read_text())["vectors"]
    if not vectors:
        raise ValueError("missing tokenizer conformance vectors")
    for vector in vectors:
        if tokenizer.encode(vector["text"], add_special_tokens=vector["add_special_tokens"]).ids != vector["ids"]:
            raise ValueError("released tokenizer vector mismatch")
    scores, frozen = fit_controls(splits["train"], tokenizer)
    validation = scores(splits["validation"])
    failures = list(gate["failures"])
    if not fixed_binding:
        failures.append("fixed source split contract is incomplete or changed")
    for name, accuracy in validation.items():
        if accuracy > 0.65:
            failures.append(name + " validation shortcut exceeds 65 percent")
    report = {"ok": gate["ok"] and not failures, "upstream": gate, "validation_controls": validation,
              "token_visible": visible, "conformance_vectors": len(vectors), "failures": failures,
               "audit_metrics": "sealed until fixed training checkpoint exists", "no_gate_bypass": True}
    report["fixed_split_contract"] = fixed_binding
    for name, sha in contract["freeze_sha256"].items():
        path = HERE / name if name == "sources.json" else root / name
        if path.is_file() and digest(path.read_bytes()) != sha:
            raise ValueError("pre-existing source/freeze bytes changed: " + name)
    if digest((HERE / "sources.json").read_bytes()) != contract["freeze_sha256"]["sources.json"]:
        raise ValueError("source manifest bytes changed")
    write_json(root / "frozen-controls.json", frozen)
    if digest((root / "frozen-controls.json").read_bytes()) != contract["freeze_sha256"]["frozen-controls.json"]:
        raise ValueError("frozen shortcut coefficients no longer reproduce")
    write_json(root / "dataset-gate.json", report)
    if "torch" in sys.modules:
        raise RuntimeError("source-only gates imported torch")
    return trainer, hp, splits, scores, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    args = parser.parse_args()
    *_, report = run_gate(args.data, args.kit)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
