# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source controls and the unchanged kit gate. Never loads a model."""

import argparse
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from collect import identities, org_number, sha, verify_bindings, write
from baselines import evaluate_frozen, load_coefficients

PIN = "1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd"


def read_rows(path):
    if path.stat().st_size > 2 << 20:
        raise ValueError("dataset file ceiling")
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def assess(observation, answer):
    """Premise: caller verified observation binding. Unknown is not VAT false."""
    if observation is None:
        return None
    try:
        if org_number(answer["org"]) != observation["org"]:
            return None
        if answer["date"] != observation["date"]:
            return None
        if (
            type(observation["registrertIMvaregisteret"]) is not bool
            or type(answer["valid"]) is not bool
        ):
            return None
        return answer["valid"] == observation["registrertIMvaregisteret"]
    except (KeyError, ValueError, TypeError):
        return None


def lookup(row):
    observation = json.loads(row["prompt"].split(". ", 2)[2])
    return assess(observation, json.loads(row["chosen"])), assess(
        observation, json.loads(row["rejected"])
    )


def controls(rows):
    observations = [json.loads(r["prompt"].split(". ", 2)[2]) for r in rows]
    missing = formatting = misleading = opposite = 0
    donors = {
        v: next(o for o in observations if o["registrertIMvaregisteret"] is v)
        for v in (False, True)
    }
    for row, obs in zip(rows, observations):
        good = json.loads(row["chosen"])
        missing += assess(None, good) is None
        formatting += (
            assess(
                obs,
                {
                    **good,
                    "org": "NO "
                    + good["org"][:3]
                    + " "
                    + good["org"][3:6]
                    + " "
                    + good["org"][6:]
                    + " MVA",
                },
            )
            is True
        )
        # A display-name distraction changes no trusted fact or assessment flag.
        misleading += (
            assess(
                obs,
                {**good, "name": donors[not obs["registrertIMvaregisteret"]]["name"]},
            )
            is True
        )
        donor = donors[not obs["registrertIMvaregisteret"]]
        # Substitute a complete real opposite-state observation. Fixed generic
        # verdict reverses; an entity-bound old answer must instead abstain.
        fixed = {"org": donor["org"], "date": donor["date"], "valid": good["valid"]}
        opposite += assess(donor, fixed) is False and assess(donor, good) is None
    return {
        "n": len(rows),
        "missing_abstentions": missing,
        "formatting_preserved": formatting,
        "misleading_display_name_ignored": misleading,
        "real_opposite_state_and_binding": opposite,
        "limits": "Constructed diagnostics on real observations, not natural vendor labels. Names never establish VAT membership.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--kit", required=True, type=Path)
    parser.add_argument("--coefficients", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    data = args.data.resolve()
    if args.report.exists() or args.report.resolve() == data / "PROBE.json":
        raise ValueError("refusing to overwrite retained source proof")
    freeze_raw = (data / "FREEZE.json").read_bytes()
    frozen = json.loads(freeze_raw)
    source = json.loads((data / "SOURCE-REPORT.json").read_text())
    if (
        sha(freeze_raw) != source["freeze_sha256"]
        or frozen["audit_opened"]
        or frozen["model_selected"]
    ):
        raise ValueError("audit freeze changed or already opened")
    all_source = read_rows(data / "observations.jsonl")
    if sha((data / "observations.jsonl").read_bytes()) != source["projection_sha256"]:
        raise ValueError("source projection hash mismatch")
    checked = verify_bindings(all_source, args.raw_root, source["probes"])
    # Check every split's bytes, but do not parse or evaluate the closed audit.
    for split, spec in frozen["files"].items():
        if sha((data / (split + ".jsonl")).read_bytes()) != spec["sha256"]:
            raise ValueError("frozen split hash mismatch")
    splits = {s: read_rows(data / (s + ".jsonl")) for s in ("train", "validation")}
    source_by_id = {r["organisasjonsnummer"]: r for r in all_source}
    seen_ids, seen_names = set(), set()
    for split, rows in splits.items():
        ids = {r["source_id"] for r in rows}
        names = {n for r in rows for n in identities(source_by_id[r["source_id"]])}
        if len(ids) != len(rows) or seen_ids & ids or seen_names & names:
            raise ValueError("entity or normalized identity overlap")
        if Counter(r["label"] for r in rows) != {
            "true": len(rows) // 2,
            "false": len(rows) // 2,
        }:
            raise ValueError("VAT preference state imbalance")
        seen_ids |= ids
        seen_names |= names
    kit = args.kit.resolve()
    path = "float_runtime/train_adapter.py"
    raw = (kit / path).read_bytes()
    pinned = subprocess.run(
        ["git", "show", PIN + ":" + path],
        cwd=kit,
        capture_output=True,
        check=True,
        timeout=20,
    ).stdout
    if raw != pinned:
        raise ValueError("kit trainer drift")
    spec = importlib.util.spec_from_file_location(
        "vendor_unchanged_trainer", kit / path
    )
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    if "torch" in sys.modules:
        raise ValueError("data-only path imported model runtime")
    # Actual gate, not copied thresholds or an AST substitute.
    gate = trainer.gate_dataset(splits["train"], trainer.Hyperparams(max_length=384))
    coefficients, coefficients_sha = load_coefficients(
        args.coefficients, frozen["files"]["train"]["sha256"]
    )
    all_baselines = evaluate_frozen(splits, coefficients)

    def subset(features):
        return {
            "train_fitted_direction": {
                n: d
                for n, d in all_baselines["train_fitted_direction"].items()
                if n in features
            },
            "by_split": {
                s: {n: v for n, v in values.items() if n in features}
                for s, values in all_baselines["by_split"].items()
            },
        }

    baselines = subset(("length", "tfidf_overlap", "alarm_keywords"))
    names = subset(("company_name_tokens", "digits"))
    exact = {
        s: sum(lookup(r) == (True, False) for r in rows) / len(rows)
        for s, rows in splits.items()
    }
    blockers = [
        "One acquisition day, not the required three disjoint dated windows.",
        "Exact source lookup already orders every inspected pair; no learned advantage established.",
        "Pinned tokenizer boundary and deployed trusted observation delivery unverified.",
        "Independent tax-authority corroboration and undeclared corporate relationships unverified.",
    ]
    if not gate["ok"]:
        blockers.append("Unchanged upstream dataset gate failed.")
    if any(
        v > 0.65
        for report in (baselines, names)
        for values in report["by_split"].values()
        for v in values.values()
    ):
        blockers.append(
            "Applicable frozen shortcut/overlap gate exceeds 65%; no bypass."
        )
    report = {
        "status": "BLOCKED_FOR_TRAINING",
        "kit_revision": PIN,
        "trainer_sha256": sha(raw),
        "raw_source_bindings": checked,
        "freeze_sha256": sha(freeze_raw),
        "upstream_dataset_gate": gate,
        "train_fitted_baselines": baselines,
        "name_and_digit_baselines": names,
        "frozen_coefficients": {
            "sha256": coefficients_sha,
            "training_sha256": coefficients["training_sha256"],
            "fit_scope": "training only; no refitting on validation or audit",
        },
        "exact_lookup_accuracy": exact,
        "controls": {s: controls(r) for s, r in splits.items()},
        "audit": "closed, hashes checked only, no model selection",
        "render_utf8_bytes_max": max(
            len(trainer.render(r["prompt"], r[side]).encode())
            for rows in splits.values()
            for r in rows
            for side in ("chosen", "rejected")
        ),
        "token_count": "unverified; byte ceiling is not token proof",
        "blockers": blockers,
        "model_loading": False,
        "training_workflow_added": False,
        "source_licence": "NLOD-2.0",
        "own_additions_licence": "LicenseRef-zkasuran-SAND-1.0",
    }
    write(args.report, report)
    print(json.dumps(report, indent=2))
    return 1


if __name__ == "__main__":
    sys.exit(main())
