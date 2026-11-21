# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only task-authoritative fit and fresh identity-quarantined audit."""
import argparse
from contextlib import contextmanager
import gc
import math
import os
from pathlib import Path

import retrain as core
import v3_remote as origin
from contracts import FAMILY, FORMATS, FOUNDATION, HERE, OWNER, file_sha, finite_bits, remote_only, strict_json, validate_ready
from development_probe import literal_wire
from remote import executor, fixed_vectors, identity_controls, score, stage_check
from v4_data import BASELINE, LABELS, QUARANTINE, pairs, prepare
from v3_data import baselines, dump

WARM = HERE / "model/SENTIMENT_ANALYSIS-endpoint-recovery.safetensors"
CONDITIONS = ("task_only", "untrusted_excerpt")


def config():
    cfg = strict_json((HERE / "v4-config.json").read_bytes())
    if cfg["source_version"] != 4 or cfg["baseline_sha256"] != file_sha(BASELINE) or cfg["quarantine_sha256"] != file_sha(QUARANTINE):
        raise ValueError("v4 source/baseline/quarantine identity changed")
    return cfg


def source(root):
    groups, manifest = prepare(root / "data")
    cfg = config()
    if file_sha(root / "data/source-manifest.json") != cfg["source_manifest_sha256"] or file_sha(WARM) != cfg["warm_package_sha256"]:
        raise ValueError("frozen v4 source or warm model changed")
    for split in ("train", "validation", "audit"):
        if manifest["files"][split]["sha256"] != cfg[split + "_sha256"]:
            raise ValueError("frozen v4 grouped split changed")
    return groups, manifest


def row_metrics(rows, wins):
    if not rows or len(rows) != len(wins) or set(row["label"] for row in rows) != set(LABELS):
        raise ValueError("complete all-label preference counts required")
    return {"accuracy": sum(wins)/len(wins), "per_label": {label: sum(win for row, win in zip(rows, wins) if row["label"] == label) /
                                                        sum(row["label"] == label for row in rows) for label in LABELS}}


def selection_key(metrics):
    subsets = metrics.get("subsets")
    if not subsets or set(subsets) != set(CONDITIONS):
        raise ValueError("both task and excerpt validation conditions required")
    values = []
    averages = []
    for formats in subsets.values():
        if set(formats) != set(FORMATS):
            raise ValueError("every condition must measure every supported format")
        for item in formats.values():
            if set(item["per_label"]) != set(LABELS):
                raise ValueError("every validation label required")
            values += list(item["per_label"].values())
            averages.append(item["accuracy"])
    if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in values + averages):
        raise ValueError("finite actual validation measurements required")
    if (type(metrics.get("pairwise_accuracy")) not in (int, float) or not math.isfinite(metrics["pairwise_accuracy"])
            or not 0 <= metrics["pairwise_accuracy"] <= 1 or type(metrics.get("val_loss")) not in (int, float)
            or not math.isfinite(metrics["val_loss"]) or metrics["val_loss"] < 0):
        raise ValueError("finite aggregate validation accuracy/loss required")
    return min(values), min(averages), metrics["pairwise_accuracy"], -metrics["val_loss"]


def evaluate(model, tokenizer, views, hp):
    remote_only()
    import torch
    from float_runtime import train_adapter as trainer
    from torch.utils.data import DataLoader
    model.eval()
    formats, subsets, losses = {}, {name: {} for name in CONDITIONS}, []
    with torch.no_grad():
        for name, rows in views.items():
            wins = []
            loader = DataLoader(rows, batch_size=hp.eval_batch, shuffle=False, collate_fn=trainer.PairCollator(tokenizer, hp.max_length))
            for batch in loader:
                chosen = model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]).logits.squeeze(-1)
                rejected = model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"]).logits.squeeze(-1)
                loss = trainer.bt_loss(chosen, rejected, batch["margin"])
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite real validation loss")
                losses.append(loss.item())
                wins.extend((chosen > rejected).tolist())
            formats[name] = row_metrics(rows, wins)
            for condition in CONDITIONS:
                kept = [(row, win) for row, win in zip(rows, wins) if row["condition"] == condition]
                subsets[condition][name] = row_metrics([row for row, _ in kept], [win for _, win in kept])
    result = {"formats": formats, "subsets": subsets, "pairwise_accuracy": sum(x["accuracy"] for x in formats.values())/len(formats),
              "val_loss": sum(losses)/len(losses)}
    selection_key(result)
    return result


def checkpoint(root, model, tokenizer, metrics, validation_pairs, manifest, hp):
    remote_only()
    core.save_checkpoint(root, model, tokenizer, metrics, validation_pairs, manifest, hp, config_fn=config, warm_path=WARM,
                         package_name="SENTIMENT_ANALYSIS-v4.safetensors",
                         selection_rule="weakest label across both task-only/untrusted-excerpt validation and all formats, then weakest subset accuracy, mean accuracy and validation loss; frozen before fresh audit")


@contextmanager
def training_policy():
    names = ("config", "source", "WARM", "pairs", "evaluate", "checkpoint", "selection_key")
    previous = {name: getattr(origin, name) for name in names}
    updated = {name: globals()[name] for name in names}
    for name, value in updated.items():
        setattr(origin, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(origin, name, value)


def train(root):
    remote_only()
    with training_policy():
        origin.train(root)


def quality_passed(quality):
    if set(quality) != set(FORMATS):
        return False
    def passed(item):
        if set(item.get("per_label", {})) != set(LABELS) or "reference_answer_only" not in item.get("shortcuts", {}):
            return False
        return (item["accuracy"] >= .8 and min(item["per_label"].values()) >= .7 and
                item["accuracy"] >= item["untrained_accuracy"] + .05 and
                item["accuracy"] >= item["shortcut_accuracy"] + .1 and
                item["shortcut_accuracy"] == max(item["shortcuts"].values()))
    return all(passed(item) and set(item["subsets"]) == set(CONDITIONS) and
               all(passed(part) for part in item["subsets"].values()) for item in quality.values())


def audit(root):
    remote_only()
    groups, manifest = source(root)
    record = strict_json((root / "training-frozen.json").read_bytes())
    if record["run_id"] != int(os.environ["GITHUB_RUN_ID"]) or record["commit"] != os.environ["GITHUB_SHA"]:
        raise ValueError("frozen model did not come from this exact run/source")
    package = root / "SENTIMENT_ANALYSIS-v4.safetensors"
    if file_sha(package) != record["package_sha256"]:
        raise ValueError("frozen candidate bytes changed")
    base = core.base_assets(root, config_fn=config)
    import numpy as np
    import torch
    from float_runtime import train_adapter as trainer
    from tokenizers import Tokenizer
    hp = trainer.Hyperparams(**config()["hyperparameters"])
    torch.set_num_threads(4)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    untrained, tokenizer = trainer.build_model(str(base), hp)
    views = {name: pairs(groups["audit"], style) for style, name in enumerate(FORMATS)}
    initial = evaluate(untrained, tokenizer, views, hp)
    del untrained
    gc.collect()
    node_tokenizer = Tokenizer.from_file(str(root / "foundation/tokenizer.json"))
    node_tokenizer.no_padding()
    node_tokenizer.no_truncation()
    tokens = {name: [node_tokenizer.encode(trainer.render(row["prompt"], row[side])).ids for row in rows for side in ("chosen", "rejected")]
              for name, rows in views.items()}
    if any(not 1 <= len(ids) <= hp.max_length for values in tokens.values() for ids in values):
        raise ValueError("audit claims truncate or exceed trained token limit")
    parsed = stage_check(root, package, record["package_sha256"], "v4-audit-parser")
    if parsed["keccak256"].removeprefix("0x") != record["package_keccak"]:
        raise ValueError("actual full-parser package identity changed")
    coefficient = strict_json(BASELINE.read_bytes())
    quality, actual, answer_actual = {}, {}, {}
    with executor(root, package, record["package_keccak"]) as (backend, runtime_row, server):
        negatives = identity_controls(server, runtime_row)
        vectors = fixed_vectors()
        first, repeated = score(backend, runtime_row, vectors), score(backend, runtime_row, vectors)
        if first != repeated:
            raise ValueError("actual reference bits are not identical")
        literal_wire(server, runtime_row, vectors, first)
        unique = {}
        answer_tokens = {}
        for name, rows in views.items():
            values = [node_tokenizer.encode(trainer.render("", row[side])).ids for row in rows for side in ("chosen", "rejected")]
            answer_tokens[name] = values
            for ids in values:
                unique.setdefault(tuple(ids), None)
        measured = score(backend, runtime_row, [list(ids) for ids in unique])
        if len(measured) != len(unique):
            raise ValueError("actual answer-only reference vector count mismatch")
        memo = dict(zip(unique, measured))
        for name, rows in views.items():
            bits = score(backend, runtime_row, tokens[name])
            answer_bits = [memo[tuple(ids)] for ids in answer_tokens[name]]
            actual[name], answer_actual[name] = bits, answer_bits
            wins = [finite_bits(bits[2*i]) > finite_bits(bits[2*i+1]) for i in range(len(rows))]
            answer_wins = [finite_bits(answer_bits[2*i]) > finite_bits(answer_bits[2*i+1]) for i in range(len(rows))]
            shortcuts = baselines(rows, coefficient)
            shortcuts["reference_answer_only"] = sum(answer_wins)/len(rows)
            item = row_metrics(rows, wins)
            item.update(untrained_accuracy=initial["formats"][name]["accuracy"],
                        untrained_scope="Actual seed-matched fresh native PyTorch adapter, not substituted reference-engine scores",
                        shortcut_accuracy=max(shortcuts.values()), shortcuts=shortcuts,
                        utf8_length_range=[min(len(trainer.render(row["prompt"], row[side]).encode()) for row in rows for side in ("chosen", "rejected")),
                                           max(len(trainer.render(row["prompt"], row[side]).encode()) for row in rows for side in ("chosen", "rejected"))],
                        token_length_range=[min(map(len, tokens[name])), max(map(len, tokens[name]))],
                        ties=sum(finite_bits(bits[2*i]) == finite_bits(bits[2*i+1]) for i in range(len(rows))), subsets={})
            for condition in CONDITIONS:
                indexes = [i for i, row in enumerate(rows) if row["condition"] == condition]
                part_rows = [rows[i] for i in indexes]
                part = row_metrics(part_rows, [wins[i] for i in indexes])
                part_shortcuts = baselines(part_rows, coefficient)
                part_shortcuts["reference_answer_only"] = sum(answer_wins[i] for i in indexes)/len(indexes)
                part.update(untrained_accuracy=initial["subsets"][condition][name]["accuracy"],
                            shortcuts=part_shortcuts, shortcut_accuracy=max(part_shortcuts.values()))
                item["subsets"][condition] = part
            quality[name] = item
    passed = quality_passed(quality)
    dump(root / "actual-quality.json", {"passed": passed, "formats": quality, "answer_only_unique_vectors": len(unique)})
    dump(root / "audit-score-bits.json", actual)
    dump(root / "answer-only-score-bits.json", answer_actual)
    if not passed:
        raise ValueError("fresh v4 audit failed unchanged quality or stronger alignment gates; no ready report")
    report = {"schema_version": 1, "status": "REGISTRATION_READY", "family": FAMILY, "owner": OWNER,
              "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"], "serving_commit": os.environ["GITHUB_SHA"],
              "package_sha256": record["package_sha256"], "package_keccak": record["package_keccak"], "package_bytes": package.stat().st_size,
              "foundation_sha256": FOUNDATION["foundation_lora_fp32.onnx"], "engine_sha256": FOUNDATION["foundation_engine.wasm"],
              "tokenizer_sha256": FOUNDATION["tokenizer.json"], "supported_labels": list(LABELS),
              "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True,
                         "source_manifest_sha256": file_sha(root / "data/source-manifest.json"), "independent_groups": len(groups["audit"]),
                         "quarantine_sha256": file_sha(QUARANTINE), "construction": manifest["construction"], "baseline_scope": manifest["baseline_scope"]},
              "quality": {"passed": True, "audit_pairs": len(views["json"]), "all_supported_formats_tested": True, "formats": quality},
              "runtime": {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
                          "negative_identity_controls": negatives, "vectors": [{"tokens": ids, "score_bits": value} for ids, value in zip(vectors, first)]},
              "training": record, "limits": [manifest["scope"], "Both preference answers may share a false excerpt; explanation truth and confidence calibration are not audited.",
                                           "Answer-only scores are actual reference-engine ablations, not universal factual proof. Source-held-out success is not canonical qualification."]}
    validate_ready(report)
    dump(root / "registration-ready.json", report)


if __name__ == "__main__":
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("train", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    try:
        {"train": train, "audit": audit}[args.phase](args.root)
    except Exception as error:
        dump(args.root / "failure.json", {"status": "FAILED", "family": FAMILY, "phase": args.phase, "run_id": os.environ["GITHUB_RUN_ID"],
                                         "commit": os.environ["GITHUB_SHA"], "error_type": type(error).__name__, "message": str(error)[:512]})
        raise
