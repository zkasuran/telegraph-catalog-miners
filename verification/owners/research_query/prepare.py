# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Freeze complete source components and non-neural train-only controls."""
import argparse
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from source import KIT, REVISION, TOKENIZER_SHA, norm, pairs, read_url, sha, write_json

HERE = Path(__file__).resolve().parent
TRAINER_SHA = "5da2919c66c2d0840104628ab88e98a9c8e5e578686f71ca9be4dcd91bd18d71"
COUNTS = {"train": 64, "validation": 16, "audit": 32}
WORDS = re.compile(r"[a-z0-9']+")


def render(prompt, answer):
    # No normalization, stripping, JSON decode or chat-template substitution.
    return f"### Task:\n{prompt}\n\n### Answer:\n{answer}"


class Components:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        self.parent[max(left, right)] = min(left, right)


def originals(report):
    if report["revision"] != REVISION or report["failures"]:
        raise ValueError("source revision or capture failure")
    rows = [c for result in report["results"] for c in result["candidates"]]
    links, seen = Components(), {}
    for row in rows:
        key = row["source_doi"]
        links.find(key)
        # DOI groups cover every sentence, page/version and both query variants.
        # Text checks union across all eligible originals, not selected rows only.
        for value in (row["source_path"], norm(row["article_title"]), norm(row["evidence"]),
                      norm(row["claim"]), norm(row["opposite_claim"])):
            if value in seen:
                links.union(key, seen[value])
            seen[value] = key
    groups = defaultdict(list)
    for row in rows:
        groups[links.find(row["source_doi"])].append(row)
    selected = [(group, min(values, key=lambda r: sha((r["source_path"] + r["evidence"]).encode()))) for group, values in groups.items()]
    return sorted(selected, key=lambda value: sha(("RESEARCH_QUERY-split-v1:" + value[0]).encode()))


def check_splits(splits):
    seen_groups, seen_doi, seen_text, seen_sources = set(), set(), set(), set()
    for name, rows in splits.items():
        groups = {r["group_id"] for r in rows}
        doi = {r["source_doi"] for r in rows}
        sources = {r["source_path"] for r in rows}
        strings = {norm(r[k]) for r in rows for k in ("evidence", "claim", "opposite_claim", "prompt", "chosen", "rejected")}
        if groups & seen_groups or doi & seen_doi or strings & seen_text or sources & seen_sources:
            raise ValueError("cross-split normalized source/entity/text overlap: " + name)
        for group in groups:
            variants = [r for r in rows if r["group_id"] == group]
            if len(variants) != 2 or {r["variant"] for r in variants} != {"original_claim", "opposite_direction"}:
                raise ValueError("mirrored original component incomplete")
        seen_groups.update(groups)
        seen_doi.update(doi)
        seen_sources.update(sources)
        seen_text.update(strings)
    return {"doi": 0, "source_page": 0, "normalized_evidence_claim_prompt_answer": 0, "component": 0}


def tokens(text):
    return WORDS.findall(text.casefold())


def features(answer):
    return [len(answer), len(tokens(answer)), answer.count("\n"), int("http" in answer),
            sum(c.isdigit() for c in answer), answer.count("Yes"), answer.count("No"),
            answer.casefold().count("agree"), answer.casefold().count("not"), answer.casefold().count("may")]


def fit_counts(train):
    # Exact integer sufficient statistics. These are symbolic shortcut counts,
    # not a neural model, floating-point optimizer or machine-dependent refit.
    vocab = Counter()
    for row in train:
        vocab.update(tokens(row["chosen"]))
        vocab.subtract(tokens(row["rejected"]))
    style = [sum(features(r["chosen"])[i] - features(r["rejected"])[i] for r in train) for i in range(10)]
    return {"schema_version": 1, "fit_split": "train", "training_rows": len(train),
            "answer_only_coefficients": dict(sorted(vocab.items())), "surface_coefficients": style,
            "length_orientation": 1, "overlap_orientation": 1,
            "algorithm": "fixed integer chosen-minus-rejected empirical feature counts, no optimizer or refit"}


def replay(rows, frozen, tokenizer):
    if not rows or frozen.get("fit_split") != "train":
        raise ValueError("empty evaluation or non-training coefficient origin")
    totals = Counter()
    for row in rows:
        c, r, p = Counter(tokens(row["chosen"])), Counter(tokens(row["rejected"])), set(tokens(row["prompt"]))
        values = {"length": len(row["chosen"]) - len(row["rejected"]),
                  "token_length": len(tokenizer.encode(render(row["prompt"], row["chosen"])).ids) - len(tokenizer.encode(render(row["prompt"], row["rejected"])).ids),
                  "overlap": sum((c[t] - r[t]) for t in p),
                  "answer_only": sum((c[t] - r[t]) * w for t, w in frozen["answer_only_coefficients"].items()),
                  "surface": sum((a - b) * w for a, b, w in zip(features(row["chosen"]), features(row["rejected"]), frozen["surface_coefficients"])),
                  "citation": int("http" in row["chosen"]) - int("http" in row["rejected"]),
                  "hedging": row["chosen"].count("may") - row["rejected"].count("may")}
        for key, value in values.items():
            totals[key] += 1 if value > 0 else .5 if value == 0 else 0
    return {key: value / len(rows) for key, value in totals.items()}


def trainer_module(source):
    if sha(source.read_bytes()) != TRAINER_SHA:
        raise ValueError("primary trainer source pin changed")
    spec = importlib.util.spec_from_file_location("research_primary_trainer", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if "torch" in sys.modules:
        raise RuntimeError("source-only gate imported a neural runtime")
    return module


def prepare(evidence, tokenizer_path, out):
    from tokenizers import Tokenizer
    if sha(tokenizer_path.read_bytes()) != TOKENIZER_SHA:
        raise ValueError("released tokenizer identity changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    report = json.loads((evidence / "source-results.json").read_text())
    pool = originals(report)
    if len(pool) < sum(COUNTS.values()):
        write_json(out / "shortage.json", {"training_allowed": False, "available_groups": len(pool), "required": COUNTS})
        raise ValueError("insufficient independent originals for fixed source recipe")
    splits, start = {}, 0
    for name, count in COUNTS.items():
        splits[name] = [row for group, original in pool[start:start + count] for row in pairs(original, group)]
        start += count
    overlap = check_splits(splits)
    lengths = []
    for rows in splits.values():
        for row in rows:
            ids = [tokenizer.encode(render(row["prompt"], row[side])).ids for side in ("chosen", "rejected")]
            lengths.extend(map(len, ids))
            if max(map(len, ids)) > 256 or ids[0] == ids[1] or len(row["prompt"].encode()) > 256:
                raise ValueError("complete decisive input or answer difference not visible")
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in splits.items():
        raw = "".join(json.dumps(r, ensure_ascii=True, separators=(",", ":")) + "\n" for r in rows).encode()
        (out / (name + ".jsonl")).write_bytes(raw)
        files[name] = {"rows": len(rows), "groups": COUNTS[name], "sha256": sha(raw)}
    frozen = fit_counts(splits["train"])
    frozen["training_sha256"] = files["train"]["sha256"]
    write_json(out / "frozen-controls.json", frozen)
    validation_controls = replay(splits["validation"], frozen, tokenizer)
    write_json(out / "data-report.json", {"schema_version": 1, "family": "RESEARCH_QUERY", "child": "RESEARCH_QUERY_GENERAL",
               "files": files, "source_revision": REVISION, "tokenizer_sha256": TOKENIZER_SHA,
               "source_sample": {"files_read": report["sample_size"], "candidate_groups": len(pool), "inventory_v1": 19699,
                                 "entire_inventory_xml_read": False}, "normalized_cross_split_overlap": overlap,
               "max_rendered_tokens": max(lengths), "max_prompt_bytes": max(len(r["prompt"].encode()) for rows in splits.values() for r in rows),
               "full_evidence_and_answers_visible": True, "audit_metrics": "sealed until checkpoint frozen",
               "label_provenance": "AI-assisted constructed preferences from original article sentences, not human-reviewed",
               "scope": "Agreement with supplied complete short scientific sentence and opposite-direction query. Not scientific truth, retrieval or broad research quality.",
               "split_method": "All eligible DOI/page/text components frozen first. SHA-256 order selects exactly 64/16/32 groups; two mirrored queries per group.",
               "external_source_holdout": False, "validation_shortcuts": validation_controls})
    trainer_path = evidence / "train_adapter.py"
    raw, rec = read_url("https://raw.githubusercontent.com/telegraphprotocol/telegraph-wasm-baseline/" + KIT + "/float_runtime/train_adapter.py", 128 << 10)
    trainer_path.write_bytes(raw)
    write_json(evidence / "trainer-receipt.json", rec)
    trainer = trainer_module(trainer_path)
    for row in splits["train"] + splits["validation"]:
        if trainer.render(row["prompt"], row["chosen"]) != render(row["prompt"], row["chosen"]):
            raise ValueError("literal primary renderer mismatch")
    hp = trainer.Hyperparams(max_length=256, lora_rank=8, lora_alpha=32, batch_size=1, grad_accum=16, epochs=3,
                            early_stopping=False, grad_checkpointing=False, eval_batch=1, gate_min_acc=.8)
    gate = trainer.gate_dataset(splits["train"], hp)
    if gate["metrics"].get("surface_probe_acc") is None:
        raise ValueError("primary surface probe did not run")
    gate["validation_controls"] = validation_controls
    gate["audit_scored"] = False
    gate["training_allowed"] = gate["ok"] and max(validation_controls.values()) <= .65
    write_json(out / "dataset-gate.json", gate)
    if not gate["training_allowed"]:
        raise ValueError("upstream or frozen validation shortcut gate failed")
    print(json.dumps({"files": files, "max_tokens": max(lengths), "gate": gate}, indent=2))
    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.evidence, args.tokenizer, args.out)
