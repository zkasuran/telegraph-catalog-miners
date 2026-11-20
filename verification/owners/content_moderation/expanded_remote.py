# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""M2 full-source preparation and bounded training/audit. Remote-only entry points."""
import argparse
import gc
import importlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from shutil import copyfile

from balanced import style_positions, training_batch, training_views
from baselines import fit as fit_baselines, scores as shortcut_scores
from contracts import FAMILY, FOUNDATION, HERE, eligibility, remote_only, sha, validate_report, write
from expanded import grouped_sources, plan, select_groups
from remote import capture_class, compile_tools, engine_audit, execute_phase, export_package, finish_report, read


def support():
    sys.path.insert(0, str(HERE / "support/expanded"))
    return {name: importlib.import_module(name) for name in ("core", "model_ops", "fit", "matched", "preserve_all")}


def prepare(root):
    remote_only()
    recipe, m = plan(), support()
    data, original, _ = m["core"].load_shared()
    original.prepare(root)
    compile_tools(root)
    capture_class(root)
    (root / "originals").mkdir()
    corpus = root / "originals/corpus"
    corpus.mkdir()
    paths = []
    for spec in recipe["files"]:
        path = corpus / Path(spec["path"]).name
        url = "https://huggingface.co/datasets/" + recipe["source_id"] + "/resolve/" + recipe["source_revision"] + "/" + spec["path"]
        data.fetch(url, path, spec["sha256"], 250 << 20)
        if path.stat().st_size != spec["bytes"]:
            raise ValueError("full source file size differs from pinned inventory")
        paths.append((spec, path))
    exclusions_raw = (HERE / "expanded-exclusions.json").read_bytes()
    if sha(exclusions_raw) != recipe["exclusions_sha256"]:
        raise ValueError("complete exposed-audit exclusions changed")
    excluded = m["core"].validate_exclusions(json.loads(exclusions_raw))
    excluded |= m["core"].validate_exclusions(json.loads(os.environ.get("CAMPAIGN_EXCLUSIONS_JSON", "[]")))
    if len(json.loads(exclusions_raw)) != 512:
        raise ValueError("one of the exposed audit groups is missing")
    from pyarrow import parquet
    by_file = {}
    def all_rows():
        for spec, path in paths:
            count = 0
            for batch in parquet.ParquetFile(path).iter_batches(batch_size=2048, columns=["text", "toxicity"]):
                for row in batch.to_pylist():
                    count += 1
                    toxicity = row["toxicity"]
                    if type(toxicity) not in (int, float) or not math.isfinite(toxicity) or not 0 <= toxicity <= 1 or not isinstance(row["text"], str):
                        raise ValueError("invalid original source annotation")
                    yield {"split": spec["split"], "text": row["text"], "label": "Safe" if toxicity == 0 else "Flagged" if toxicity >= 0.8 else None}
            by_file[spec["path"]] = count
    rows, inventory = grouped_sources(all_rows(), excluded, data.group_id, recipe["max_source_bytes"])
    from tokenizers import Tokenizer
    if sha((root / "foundation/tokenizer.json").read_bytes()) != FOUNDATION["tokenizer.json"]:
        raise ValueError("source-only tokenizer differs from the node")
    tokenizer = Tokenizer.from_file(str(root / "foundation/tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    def admissible(row, split):
        raw = data.pair(FAMILY, row)
        views = training_views([raw], data.pair).values() if split == "train" else [[data.pair(FAMILY, row, style)] for style in range(3)]
        return all(len(tokenizer.encode("### Task:\n" + pair["prompt"] + "\n\n### Answer:\n" + pair[side]).ids) <= recipe["max_tokens"]
                   for values in views for pair in values for side in ("chosen", "rejected"))
    sizes = {name: recipe[name + "_groups"] for name in ("train", "validation", "audit")}
    selected, length_rejected = select_groups(rows, sizes, recipe["selection_salts"], admissible)
    del rows
    splits = {name: [data.pair(FAMILY, row) for row in values] for name, values in selected.items()}
    splits["train"] = training_batch(splits["train"], style_positions(splits["train"]), 0, data.pair)
    data.check_splits(splits)
    source_files = {}
    for name in ("train", "validation"):
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in splits[name]).encode()
        (root / "originals" / (name + ".jsonl")).write_bytes(raw)
        source_files[name] = {"sha256": sha(raw), "n": len(splits[name])}
    audit_raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in splits["audit"]).encode()
    (root / "audit-private").mkdir()
    (root / "audit-private/audit.jsonl").write_bytes(audit_raw)
    freeze = {"source_files": source_files, "audit_sha256": sha(audit_raw), "audit_groups": [r["group_id"] for r in splits["audit"]],
              "audit_labels": dict(Counter(r["label"] for r in splits["audit"])), "normalized_overlap": 0, "exposed_audit_overlap": 0,
              "source": data.SOURCES[FAMILY], "full_inventory": recipe["files"], "source_inventory": {**inventory, "by_file": by_file},
              "source_only_token_ceiling_rejected": length_rejected, "plan_sha256": sha((HERE / "expanded.json").read_bytes()), "frozen_before_model_loading": True}
    write(root / "evidence/audit-freeze.json", freeze)
    warm = recipe["warm_start"]
    for name, tensor, config, expected in (("training-start", warm["checkpoint"], warm["config"], warm["checkpoint_sha256"]),
                                           ("untrained", warm["untrained"], warm["untrained_config"], warm["untrained_sha256"])):
        if sha((HERE / "assets" / tensor).read_bytes()) != expected or sha((HERE / "assets" / config).read_bytes()) != warm["config_sha256"]:
            raise ValueError("licensed M2 initialization changed")
        target = root / "adapters" / name
        target.mkdir(parents=True)
        copyfile(HERE / "assets" / tensor, target / "adapter_model.safetensors")
        copyfile(HERE / "assets" / config, target / "adapter_config.json")
    views = training_views(splits["train"], data.pair)
    coefficients = fit_baselines([row for values in views.values() for row in values])
    write(root / "evidence/baseline-coefficients.json", coefficients)
    shortcuts = {name: shortcut_scores(values, coefficients) for name, values in views.items()}
    write(root / "evidence/training-shortcut-gates.json", {"scores": shortcuts, "ceiling": 0.65, "scope": "Training only"})
    if any(max(row.values()) > 0.65 for row in shortcuts.values()):
        raise ValueError("M2 training shortcut source gate failed")
    write(root / "evidence/source-manifest.json", {"sources": freeze, "coefficient_sha256": sha((root / "evidence/baseline-coefficients.json").read_bytes()),
          "scope": recipe["scope"], "preference_authorship": recipe["construction"], "support": read(HERE / "SUPPORT.json")})


def initial(root):
    remote_only()
    m = support()
    torch, np, trainer, hp = m["model_ops"].runtime(root, 2)
    splits = m["model_ops"].read_development(root)
    gate = trainer.gate_dataset(splits["train"], hp)
    write(root / "evidence/dataset-gate.json", gate)
    if not gate["ok"]:
        raise ValueError("M2 original dataset gate failed before model loading")
    write(root / "evidence/rng-controls.json", importlib.import_module("remote_controls").controls())
    tok, node = m["model_ops"].tokenizer(root)
    views = m["model_ops"].views(splits["validation"])
    m["model_ops"].tokenize(trainer, tok, node, [splits["train"], splits["validation"], *views.values()], root / "evidence/tokenizer-check.json")
    m["model_ops"].seed(torch, np, hp)
    model, tok = trainer.build_model(str(root / "base"), hp)
    m["model_ops"].reload_snapshot(root, "untrained", model)
    with m["core"].isolated_rng(torch, np):
        baseline = m["model_ops"].score_views(trainer, model, tok, views, hp)
        saved = m["model_ops"].save_snapshot(root, "untrained", model, tok, trainer)
    write(root / "evidence/untrained-validation.json", {"formats": baseline, **saved, "scope": "Genuine untrained H9 initialization, not fitted B3 weights"})
    m["model_ops"].reload_snapshot(root, "training-start", model)
    m["model_ops"].seed(torch, np, hp)
    frozen = m["matched"].freeze_initial(root, model, torch, np, splits["train"])
    with m["core"].isolated_rng(torch, np):
        saved = m["model_ops"].save_snapshot(root, "training-start", model, tok, trainer)
    write(root / "evidence/training-start.json", {**saved, "initial_state_sha256": frozen["initial_state_sha256"], "scope": "Validation-selected B3 epoch 1, fresh optimizer, no audit-driven selection"})


def select(root):
    remote_only()
    m = support()
    m["core"].deny_measurement_reads(root)
    matching = m["matched"].matched_report(root)
    restoration = read(root / "evidence/M2-restoration.json")
    if not restoration["parameter_bytes_equal"] or not restoration["validation_replay_equal"] or not read(root / "evidence/M2-dataset-gate.json")["ok"]:
        raise ValueError("M2 source/restoration gates failed")
    coefficients = read(root / "evidence/baseline-coefficients.json")
    if sha((root / "evidence/baseline-coefficients.json").read_bytes()) != read(root / "evidence/source-manifest.json")["coefficient_sha256"]:
        raise ValueError("frozen training coefficients changed")
    views = m["model_ops"].views(m["model_ops"].read_development(root)["validation"])
    shortcuts = {name: max(shortcut_scores(values, coefficients).values()) for name, values in views.items()}
    untrained = read(root / "evidence/untrained-validation.json")
    records = read(root / "evidence/M2-validation-history.json")
    if not records or len(records) > 2:
        raise ValueError("bounded actual M2 history required")
    eligible = [r for r in records if not eligibility(r["formats"], untrained["formats"], shortcuts)]
    write(root / "evidence/selection.json", {"matching": matching, "shortcuts": shortcuts, "selection_uses_audit": False,
          "transcript": [{"epoch": r["epoch"], "key": m["core"].rank_key(r), "failures": eligibility(r["formats"], untrained["formats"], shortcuts)} for r in records]})
    if not eligible:
        write(root / "evidence/result.json", {"status": "NO_VALIDATION_CANDIDATE", "audit_opened": False})
        raise ValueError("M2 validation failed; fresh audit stays sealed")
    candidate = min(eligible, key=m["core"].rank_key)
    for row in (candidate, untrained):
        if sha((root / "evidence" / (row["snapshot"] + ".safetensors")).read_bytes()) != row["checkpoint_sha256"]:
            raise ValueError("selected M2 checkpoint changed")
    write(root / "evidence/candidate-freeze.json", {"candidate": candidate, "untrained": untrained,
          "source_manifest_sha256": sha((root / "evidence/source-manifest.json").read_bytes()), "selection_sha256": sha((root / "evidence/selection.json").read_bytes()),
          "experiment_sha256": sha((HERE / "expanded.json").read_bytes()), "audit_sha256": read(root / "evidence/audit-freeze.json")["audit_sha256"]})


def qualify(root):
    remote_only()
    m = support()
    frozen = read(root / "evidence/candidate-freeze.json")
    if frozen["experiment_sha256"] != sha((HERE / "expanded.json").read_bytes()) or frozen["source_manifest_sha256"] != sha((root / "evidence/source-manifest.json").read_bytes()) or frozen["selection_sha256"] != sha((root / "evidence/selection.json").read_bytes()):
        raise ValueError("frozen M2 experiment or source changed")
    torch, np, trainer, hp = m["model_ops"].runtime(root, 2)
    tok, node = m["model_ops"].tokenizer(root)
    m["model_ops"].seed(torch, np, hp)
    model, tok = trainer.build_model(str(root / "base"), hp)
    m["model_ops"].reload_snapshot(root, frozen["candidate"]["snapshot"], model)
    with m["core"].isolated_rng(torch, np):
        validation = m["model_ops"].score_views(trainer, model, tok, m["model_ops"].views(m["model_ops"].read_development(root)["validation"]), hp)
    if validation != frozen["candidate"]["formats"]:
        raise ValueError("actual M2 validation margins did not replay")
    del model
    gc.collect()
    trained, trained_manifest = export_package(root, FAMILY + ".safetensors", frozen["candidate"])
    untrained, untrained_manifest = export_package(root, "UNTRAINED.safetensors", frozen["untrained"])
    with (root / "evidence/audit-access.json").open("x") as marker:
        marker.write(json.dumps({"frozen_candidate_sha256": frozen["candidate"]["checkpoint_sha256"], "audit_sha256": frozen["audit_sha256"], "measurement": "One fresh independent audit, never selection"}) + "\n")
    raw = (root / "audit-private/audit.jsonl").read_bytes()
    if sha(raw) != frozen["audit_sha256"]:
        raise ValueError("M2 fresh audit changed")
    data, _, _ = m["core"].load_shared()
    originals = data.decode_jsonl(raw)
    if len(originals) != 256 or len({r["group_id"] for r in originals}) != 256:
        raise ValueError("M2 independent audit coverage changed")
    data.check_splits({**m["model_ops"].read_development(root), "audit": originals})
    if set(read(HERE / "expanded-exclusions.json")) & {r["group_id"] for r in originals}:
        raise ValueError("an exposed audit group entered M2 measurement")
    views = m["model_ops"].views(originals)
    m["model_ops"].tokenize(trainer, tok, node, list(views.values()), root / "evidence/audit-tokenizer-check.json")
    coefficients = read(root / "evidence/baseline-coefficients.json")
    if sha((root / "evidence/baseline-coefficients.json").read_bytes()) != read(root / "evidence/source-manifest.json")["coefficient_sha256"]:
        raise ValueError("immutable M2 coefficients changed")
    shortcuts = {name: max(shortcut_scores(values, coefficients).values()) for name, values in views.items()}
    tokens = {name: [node.encode(trainer.render(r["prompt"], r[side])).ids for r in values for side in ("chosen", "rejected")] for name, values in views.items()}
    trained_bits, vectors, negative = engine_audit(root, trained, trained_manifest, tokens)
    untrained_bits, _, _ = engine_audit(root, untrained, untrained_manifest, tokens)
    finish_report(root, frozen, trained, trained_manifest, originals, views, trained_bits, untrained_bits, shortcuts, vectors, negative,
                  {"kind": "M2 broader-source model with fresh independent audit", "experiment_sha256": frozen["experiment_sha256"],
                   "audit_sha256": frozen["audit_sha256"], "exposed_audit_overlap": 0, "warm_start": plan()["warm_start"]})
    validate_report(read(root / "evidence/registration-ready.json"))
    result_files = ["registration-ready.json", FAMILY + ".safetensors", FAMILY + ".safetensors.parser.json", FAMILY + ".safetensors.score-bits.json",
                    "UNTRAINED.safetensors.parser.json", "UNTRAINED.safetensors.score-bits.json", "source-manifest.json", "baseline-coefficients.json",
                    "candidate-freeze.json", "selection.json", "class-snapshot.json", "audit-tokenizer-check.json", "audit-freeze.json", "training-format-gates.json"]
    if sum((root / "evidence" / name).stat().st_size for name in result_files) > 15 << 20:
        raise ValueError("actual M2 registration evidence exceeds bounded archive budget")


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("run", "prepare", "initial", "select", "qualify"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase != "run":
        {"prepare": prepare, "initial": initial, "select": select, "qualify": qualify}[args.phase](root)
        return
    if root.exists():
        raise ValueError("fresh M2 run root required")
    (root / "evidence").mkdir(parents=True)
    write(root / "evidence/run-binding.json", {"family": FAMILY, "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
          "experiment_sha256": sha((HERE / "expanded.json").read_bytes()), "support": read(HERE / "SUPPORT.json")})
    deadline = time.monotonic() + 17400
    command = [sys.executable, str(HERE / "expanded_remote.py")]
    try:
        execute_phase(root, "prepare", [*command, "prepare", "--root", str(root)], 1800, deadline)
        execute_phase(root, "initial", [*command, "initial", "--root", str(root)], 1800, deadline)
        execute_phase(root, "fit-M2", [sys.executable, str(HERE / "support/expanded/fit.py"), "--arm", "M2", "--root", str(root)], 9000, deadline)
        execute_phase(root, "select", [*command, "select", "--root", str(root)], 120, deadline)
        execute_phase(root, "qualify", [*command, "qualify", "--root", str(root)], 12600, deadline)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
        write(root / "evidence/operational-failure.json", {"status": "FAILED", "error_type": type(error).__name__, "release_gate": "INCOMPLETE"})
        raise
    finally:
        support()["preserve_all"].preserve(root)


if __name__ == "__main__":
    main()
