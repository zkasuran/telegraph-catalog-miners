# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded format-balanced fitting with a clean audit frozen before model work."""
import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from shutil import copyfile

from balanced import audit_select, check_audit_identity, plan, training_views
from baselines import fit as fit_baselines, scores as shortcut_scores
from contracts import FAMILY, HERE, eligibility, remote_only, sha, validate_report, write
from remote import capture_class, compile_tools, engine_audit, execute_phase, export_package, finish_report, initial as original_initial, read, support


def prepare(root):
    remote_only()
    recipe = plan()
    m = support()
    data, original, _ = m["core"].load_shared()
    original.prepare(root)
    compile_tools(root)
    capture_class(root)
    originals_path = root / "originals"
    originals_path.mkdir()
    selected, removed = data.select(data.moderation(originals_path))
    splits = {name: [data.pair(FAMILY, row) for row in rows] for name, rows in selected.items()}
    data.check_splits(splits)
    exclusions_raw = (HERE / "excluded-audit-groups.json").read_bytes()
    if sha(exclusions_raw) != recipe["excluded_audit_groups_sha256"]:
        raise ValueError("complete exposed audit exclusion file changed")
    exposed = m["core"].validate_exclusions(json.loads(exclusions_raw))
    if len(exposed) != 256:
        raise ValueError("missing exposed audit groups")
    extra = m["core"].validate_exclusions(json.loads(os.environ.get("CAMPAIGN_EXCLUSIONS_JSON", "[]")))
    excluded = set(exposed) | extra
    source_files = {}
    for name, rows in splits.items():
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        if sha(raw) != m["core"].CONTRACT["original_splits"][name]:
            raise ValueError("original development source changed")
        (originals_path / (name + ".jsonl")).write_bytes(raw)
        excluded.update(row["group_id"] for row in rows)
        source_files[name] = {"sha256": sha(raw), "n": len(rows)}
    from pyarrow import parquet
    scan = {"raw_rows": 0}
    def official_rows():
        for batch in parquet.ParquetFile(originals_path / "test.parquet").iter_batches(batch_size=2048, columns=["text", "toxicity"]):
            for row in batch.to_pylist():
                scan["raw_rows"] += 1
                if scan["raw_rows"] > 300000:
                    raise ValueError("complete official file scan ceiling")
                text, value = row["text"].strip(), row["toxicity"]
                if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("invalid official source annotation")
                if not 8 <= len(text.encode()) <= data.MAX_TEXT_BYTES:
                    continue
                group = data.group_id(text)
                yield {"text": text, "label": "Safe" if value == 0 else "Flagged" if value >= 0.8 else None,
                       "group_id": group, "source_id": group, "domain": "civil_comments", "split": "test"}
    audit, inventory = audit_select(official_rows(), excluded, recipe["audit_salt"], recipe["audit_per_label"])
    audit_rows = [data.pair(FAMILY, row) for row in audit]
    data.check_splits({"train": splits["train"], "validation": splits["validation"], "audit": audit_rows})
    if set(r["group_id"] for r in audit_rows) & exposed:
        raise ValueError("exposed audit source reused")
    audit_raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit_rows).encode()
    check_audit_identity(audit_raw, [r["group_id"] for r in audit_rows], recipe)
    (root / "audit-private").mkdir()
    (root / "audit-private/audit.jsonl").write_bytes(audit_raw)
    freeze = {"audit_sha256": sha(audit_raw), "audit_groups": [r["group_id"] for r in audit_rows],
              "audit_labels": dict(Counter(r["label"] for r in audit_rows)), "exposed_audit_overlap": 0,
              "normalized_overlap": 0, "source_files": source_files, "source": data.SOURCES[FAMILY],
              "inventory": {**inventory, "raw_official_file_rows": scan["raw_rows"], "scan_scope": "Complete pinned official test file; scanned_rows counts text-length-admitted rows"}, "removed_original_occurrences": removed,
              "excluded_groups_sha256": sha(json.dumps(sorted(excluded)).encode()), "excluded_count": len(excluded),
              "plan_sha256": sha((HERE / "balanced.json").read_bytes()), "frozen_before_model_loading": True}
    write(root / "evidence/audit-freeze.json", freeze)
    warm = recipe["warm_start"]
    for name, tensor, config, expected in (("training-start", warm["checkpoint"], warm["config"], warm["checkpoint_sha256"]),
                                           ("untrained", warm["untrained"], warm["untrained_config"], warm["untrained_sha256"])):
        if sha((HERE / "assets" / tensor).read_bytes()) != expected or sha((HERE / "assets" / config).read_bytes()) != warm["config_sha256"]:
            raise ValueError("licensed fitted/untrained input changed")
        directory = root / "adapters" / name
        directory.mkdir(parents=True)
        copyfile(HERE / "assets" / tensor, directory / "adapter_model.safetensors")
        copyfile(HERE / "assets" / config, directory / "adapter_config.json")
    anchored_views = training_views(splits["train"], data.pair)
    coefficients = fit_baselines([r for rows in anchored_views.values() for r in rows])
    write(root / "evidence/baseline-coefficients.json", coefficients)
    training_shortcuts = {name: shortcut_scores(rows, coefficients) for name, rows in anchored_views.items()}
    write(root / "evidence/training-shortcut-gates.json", {"scores": training_shortcuts, "ceiling": 0.65,
          "coefficient_sha256": sha((root / "evidence/baseline-coefficients.json").read_bytes()), "scope": "Train-only source gate, not validation or audit"})
    if any(max(values.values()) > 0.65 for values in training_shortcuts.values()):
        raise ValueError("strongest training-format shortcut exceeds source gate")
    write(root / "evidence/source-manifest.json", {"sources": freeze, "coefficient_sha256": sha((root / "evidence/baseline-coefficients.json").read_bytes()),
          "training_format_policy": recipe["training_format_policy"], "scope": data.SOURCES[FAMILY]["scope"],
          "preference_authorship": "AI-assisted construction from original public annotations, not human-reviewed preferences", "support": read(HERE / "SUPPORT.json")})


def initial(root):
    remote_only()
    original_initial(root)
    value = read(root / "evidence/training-start.json")
    value["scope"] = "Licensed F3 epoch 1 warm start, fresh optimizer. Original failed audit is excluded, never a training input."
    write(root / "evidence/training-start.json", value)


def select(root):
    remote_only()
    m = support()
    m["core"].deny_measurement_reads(root)
    arm = plan()["arm"]
    matching = m["matched"].matched_report(root)
    restoration = read(root / "evidence" / (arm + "-restoration.json"))
    if not restoration["parameter_bytes_equal"] or not restoration["validation_replay_equal"] or not read(root / "evidence/dataset-gate.json")["ok"] or not read(root / "evidence" / (arm + "-dataset-gate.json"))["ok"]:
        raise ValueError("real source/restoration controls failed")
    coefficients = read(root / "evidence/baseline-coefficients.json")
    source = read(root / "evidence/source-manifest.json")
    if sha((root / "evidence/baseline-coefficients.json").read_bytes()) != source["coefficient_sha256"]:
        raise ValueError("immutable train-fitted coefficients changed")
    views = m["model_ops"].views(m["model_ops"].read_development(root)["validation"])
    shortcuts = {name: max(shortcut_scores(rows, coefficients).values()) for name, rows in views.items()}
    untrained = read(root / "evidence/untrained-validation.json")
    history = read(root / "evidence" / (arm + "-validation-history.json"))
    eligible = [r for r in history if not eligibility(r["formats"], untrained["formats"], shortcuts)]
    write(root / "evidence/selection.json", {"matching": matching, "selection_uses_audit": False, "shortcuts": shortcuts,
          "transcript": [{"epoch": r["epoch"], "failures": eligibility(r["formats"], untrained["formats"], shortcuts), "key": m["core"].rank_key(r)} for r in history]})
    if not eligible:
        write(root / "evidence/result.json", {"status": "NO_VALIDATION_CANDIDATE", "audit_opened": False})
        raise ValueError("no validation-eligible balanced candidate; clean audit remains sealed")
    candidate = min(eligible, key=m["core"].rank_key)
    for row in (candidate, untrained):
        if sha((root / "evidence" / (row["snapshot"] + ".safetensors")).read_bytes()) != row["checkpoint_sha256"]:
            raise ValueError("selected checkpoint changed")
    write(root / "evidence/candidate-freeze.json", {"candidate": candidate, "untrained": untrained,
          "source_manifest_sha256": sha((root / "evidence/source-manifest.json").read_bytes()), "selection_sha256": sha((root / "evidence/selection.json").read_bytes()),
          "experiment_sha256": sha((HERE / "balanced.json").read_bytes()), "audit_sha256": read(root / "evidence/audit-freeze.json")["audit_sha256"]})


def qualify(root):
    remote_only()
    m = support()
    frozen = read(root / "evidence/candidate-freeze.json")
    if frozen["experiment_sha256"] != sha((HERE / "balanced.json").read_bytes()) or frozen["source_manifest_sha256"] != sha((root / "evidence/source-manifest.json").read_bytes()) or frozen["selection_sha256"] != sha((root / "evidence/selection.json").read_bytes()):
        raise ValueError("frozen balanced selection changed")
    torch, np, trainer, hp = m["model_ops"].runtime(root, 3)
    tok, node = m["model_ops"].tokenizer(root)
    m["model_ops"].seed(torch, np, hp)
    model, tok = trainer.build_model(str(root / "base"), hp)
    m["model_ops"].reload_snapshot(root, frozen["candidate"]["snapshot"], model)
    with m["core"].isolated_rng(torch, np):
        validation = m["model_ops"].score_views(trainer, model, tok, m["model_ops"].views(m["model_ops"].read_development(root)["validation"]), hp)
    if validation != frozen["candidate"]["formats"]:
        raise ValueError("actual validation replay changed")
    del model
    gc.collect()
    trained, trained_manifest = export_package(root, FAMILY + ".safetensors", frozen["candidate"])
    untrained, untrained_manifest = export_package(root, "UNTRAINED.safetensors", frozen["untrained"])
    with (root / "evidence/audit-access.json").open("x") as marker:
        marker.write(json.dumps({"frozen_candidate_sha256": frozen["candidate"]["checkpoint_sha256"], "fresh_audit_sha256": frozen["audit_sha256"], "measurement": "One independent audit, never selection"}) + "\n")
    raw = (root / "audit-private/audit.jsonl").read_bytes()
    if sha(raw) != frozen["audit_sha256"]:
        raise ValueError("fresh independent audit changed")
    data, _, _ = m["core"].load_shared()
    originals = data.decode_jsonl(raw)
    if len(originals) != 256 or len({r["group_id"] for r in originals}) != 256:
        raise ValueError("fresh independent audit count changed")
    data.check_splits({**m["model_ops"].read_development(root), "audit": originals})
    excluded = set(read(HERE / "excluded-audit-groups.json"))
    if excluded & {r["group_id"] for r in originals}:
        raise ValueError("old exposed audit enters new measurement")
    views = m["model_ops"].views(originals)
    m["model_ops"].tokenize(trainer, tok, node, list(views.values()), root / "evidence/audit-tokenizer-check.json")
    coefficients = read(root / "evidence/baseline-coefficients.json")
    if sha((root / "evidence/baseline-coefficients.json").read_bytes()) != read(root / "evidence/source-manifest.json")["coefficient_sha256"]:
        raise ValueError("train-fitted coefficients changed after freeze")
    shortcuts = {name: max(shortcut_scores(rows, coefficients).values()) for name, rows in views.items()}
    tokens = {name: [node.encode(trainer.render(r["prompt"], r[side])).ids for r in rows for side in ("chosen", "rejected")] for name, rows in views.items()}
    trained_bits, vectors, negative = engine_audit(root, trained, trained_manifest, tokens)
    untrained_bits, _, _ = engine_audit(root, untrained, untrained_manifest, tokens)
    finish_report(root, frozen, trained, trained_manifest, originals, views, trained_bits, untrained_bits, shortcuts, vectors, negative,
                  {"kind": "New format-balanced training candidate with clean independent audit", "experiment_sha256": frozen["experiment_sha256"],
                   "audit_sha256": frozen["audit_sha256"], "exposed_audit_overlap": 0, "warm_start": plan()["warm_start"]})
    validate_report(read(root / "evidence/registration-ready.json"))
    bounded_assets = ["registration-ready.json", FAMILY + ".safetensors", FAMILY + ".safetensors.parser.json", FAMILY + ".safetensors.score-bits.json",
                      "UNTRAINED.safetensors.parser.json", "UNTRAINED.safetensors.score-bits.json", "source-manifest.json", "baseline-coefficients.json",
                      "candidate-freeze.json", "selection.json", "class-snapshot.json", "audit-tokenizer-check.json", "audit-freeze.json", "training-format-gates.json"]
    if sum((root / "evidence" / name).stat().st_size for name in bounded_assets) > 15 << 20:
        raise ValueError("actual registration asset bytes exceed bounded artifact budget")


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
        raise ValueError("fresh format-balanced run root required")
    (root / "evidence").mkdir(parents=True)
    write(root / "evidence/run-binding.json", {"family": FAMILY, "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
          "experiment_sha256": sha((HERE / "balanced.json").read_bytes()), "support": read(HERE / "SUPPORT.json")})
    deadline = time.monotonic() + 17400
    command = [sys.executable, str(HERE / "balanced_remote.py")]
    try:
        execute_phase(root, "prepare", [*command, "prepare", "--root", str(root)], 1800, deadline)
        execute_phase(root, "initial", [*command, "initial", "--root", str(root)], 1200, deadline)
        execute_phase(root, "fit-B3", [sys.executable, str(HERE / "support/moderation-diagnostic/fit.py"), "--arm", "B3", "--root", str(root)], 9000, deadline)
        execute_phase(root, "select", [*command, "select", "--root", str(root)], 120, deadline)
        execute_phase(root, "qualify", [*command, "qualify", "--root", str(root)], 12600, deadline)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
        write(root / "evidence/operational-failure.json", {"status": "FAILED", "error_type": type(error).__name__, "release_gate": "INCOMPLETE"})
        raise
    finally:
        support()["preserve_all"].preserve(root)


if __name__ == "__main__":
    main()
