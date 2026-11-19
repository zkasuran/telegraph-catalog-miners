# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Separate initial-control, validation-selection and frozen-audit processes."""
import argparse
import json
from pathlib import Path

from core import (
    CONTRACT,
    FORMATS,
    LABELS,
    choose,
    deny_measurement_reads,
    isolated_rng,
    load_shared,
    quality,
    rank_key,
    remote_only,
    sha,
    write,
)
from model_ops import (
    read_development,
    reload_snapshot,
    runtime,
    save_snapshot,
    score_views,
    seed,
    tokenize,
    tokenizer,
    views,
)


def initial(root):
    deny_measurement_reads(root)
    torch, np, trainer, hp = runtime(root, 3)
    splits = read_development(root)
    gate = trainer.gate_dataset(splits["train"], hp)
    write(root / "evidence" / "dataset-gate.json", gate)
    if not gate["ok"]:
        raise RuntimeError("upstream dataset gate failed before model loading")
    from remote_controls import controls
    write(root / "evidence" / "rng-controls.json", controls())
    tok, node = tokenizer(root)
    formats = views(splits["validation"])
    tokenize(trainer, tok, node, [splits["train"], splits["validation"], *formats.values()], root / "evidence" / "tokenizer-check.json")
    seed(torch, np, hp)
    model, tok = trainer.build_model(str(root / "base"), hp)
    with isolated_rng(torch, np):
        untrained = score_views(trainer, model, tok, formats, hp)
        snapshot = save_snapshot(root, "untrained", model, tok, trainer)
    write(root / "evidence" / "untrained-validation.json", {"formats": untrained, **snapshot})


def select(root):
    deny_measurement_reads(root)
    evidence = root / "evidence"
    if (evidence / "candidate-freeze.json").exists() or (evidence / "audit-access.json").exists():
        raise ValueError("candidate selection is immutable")
    records = []
    for arm in CONTRACT["arms"]:
        # A missing completed arm or restoration control invalidates comparison.
        restoration = json.loads((evidence / (arm + "-restoration.json")).read_text())
        if not restoration["parameter_bytes_equal"] or not restoration["validation_replay_equal"]:
            raise ValueError("missing arm restoration gate")
        records.extend(json.loads((evidence / (arm + "-validation-history.json")).read_text()))
    if len(records) > 9:
        raise ValueError("snapshot ceiling exceeded")
    baseline = json.loads((evidence / "untrained-validation.json").read_text())
    dataset_ok = json.loads((evidence / "dataset-gate.json").read_text())["ok"]
    chosen = choose(records, baseline["formats"], dataset_ok)
    transcript = [{"arm": r["arm"], "epoch": r["epoch"], "rank_key": rank_key(r),
                   "failures": quality(r["formats"], baseline["formats"], dataset_ok)} for r in records]
    c3 = min((r for r in records if r["arm"] == "C3"), key=rank_key)
    h6 = min((r for r in records if r["arm"] == "H6"), key=rank_key)
    def cells(r):
        return [r["formats"][f]["by_label"][label]["accuracy"] for f in FORMATS for label in LABELS]
    c, h = cells(c3), cells(h6)
    improvement = min(h) + 1e-12 >= min(c) + 0.05 and all(b >= a for a, b in zip(c, h))
    write(evidence / "selection.json", {"eligible": chosen is not None, "transcript": transcript,
           "h6_all_label_improvement_over_c3": improvement, "selection_uses_audit": False})
    if chosen is None:
        write(evidence / "result.json", {"verdict": "FAIL", "phase": "validation_selection",
              "audit_opened": False, "parser": "not attempted", "runtime": "not attempted", "node": "not attempted"})
        raise RuntimeError("no all-label validation-eligible snapshot; audit remains unopened")
    freeze = json.loads((evidence / "audit-freeze.json").read_text())
    for name, expected in ((chosen["snapshot"], chosen["checkpoint_sha256"]), ("untrained", baseline["checkpoint_sha256"])):
        if sha((evidence / (name + ".safetensors")).read_bytes()) != expected:
            raise ValueError("candidate or untrained checkpoint changed before freeze")
    write(evidence / "candidate-freeze.json", {"candidate": chosen, "untrained": baseline,
          "audit_sha256": freeze["audit_sha256"], "audit_freeze_sha256": sha((evidence / "audit-freeze.json").read_bytes()),
          "selection_sha256": sha((evidence / "selection.json").read_bytes()),
          "contract_sha256": sha((Path(__file__).parent / "contract.json").read_bytes()),
          "parser": "not attempted", "runtime": "not attempted", "node": "not attempted"})


def final(root):
    # Only this process can open audit data after verifying the candidate freeze.
    evidence = root / "evidence"
    frozen = json.loads((evidence / "candidate-freeze.json").read_text())
    if sha((Path(__file__).parent / "contract.json").read_bytes()) != frozen["contract_sha256"]:
        raise ValueError("diagnostic contract changed")
    if sha((evidence / "audit-freeze.json").read_bytes()) != frozen["audit_freeze_sha256"]:
        raise ValueError("audit freeze changed")
    if sha((evidence / "selection.json").read_bytes()) != frozen["selection_sha256"]:
        raise ValueError("selector transcript changed")
    for row in (frozen["candidate"], frozen["untrained"]):
        name = row["snapshot"]
        for path, expected in ((evidence / (name + ".safetensors"), row["checkpoint_sha256"]),
                               (root / "adapters" / name / "adapter_model.safetensors", row["adapter_sha256"]),
                               (root / "adapters" / name / "adapter_config.json", row["config_sha256"])):
            if sha(path.read_bytes()) != expected:
                raise ValueError("frozen candidate/control identity changed")
    torch, np, trainer, hp = runtime(root, 3)
    splits = read_development(root)
    tok, node = tokenizer(root)
    seed(torch, np, hp)
    model, tok = trainer.build_model(str(root / "base"), hp)
    restored = reload_snapshot(root, frozen["candidate"]["snapshot"], model)
    with isolated_rng(torch, np):
        replay = score_views(trainer, model, tok, views(splits["validation"]), hp)
    if replay != frozen["candidate"]["formats"]:
        raise ValueError("frozen candidate validation replay changed before audit access")
    marker = evidence / "audit-access.json"
    with marker.open("x") as out:
        out.write(json.dumps({"candidate_freeze_sha256": sha((evidence / "candidate-freeze.json").read_bytes()),
                              "measurement": "one final candidate/control pass; audit may not be rerun"}) + "\n")
    raw = (root / "audit-private" / "audit.jsonl").read_bytes()
    if sha(raw) != frozen["audit_sha256"]:
        raise ValueError("audit file changed")
    data, _, _ = load_shared()
    audit = [{**r, "margin": 0.0} for r in data.decode_jsonl(raw)]
    if len(audit) != 256:
        raise ValueError("audit original count changed")
    data.check_splits({**splits, "audit": audit})
    excluded = set(json.loads((evidence / "exclusions.json").read_text()))
    manifest = json.loads((evidence / "audit-freeze.json").read_text())
    if sha(json.dumps(sorted(excluded), separators=(",", ":")).encode()) != manifest["exclusion_sha256"]:
        raise ValueError("exclusion manifest changed")
    if [r["group_id"] for r in audit] != manifest["audit_groups"]:
        raise ValueError("audit group identity changed")
    if excluded.intersection(r["group_id"] for r in audit):
        raise ValueError("audit exclusion overlap")
    formats = views(audit)
    tokenize(trainer, tok, node, list(formats.values()), evidence / "audit-tokenizer-check.json")
    with isolated_rng(torch, np):
        trained = score_views(trainer, model, tok, formats, hp)
    # Fixed adapter from the already frozen seed-matched initialization.
    reload_snapshot(root, "untrained", model)
    with isolated_rng(torch, np):
        untrained = score_views(trainer, model, tok, formats, hp)
    gate = json.loads((evidence / "dataset-gate.json").read_text())
    failures = quality(trained, untrained, gate["ok"])
    write(evidence / "result.json", {"verdict": "TRAINED_CANDIDATE" if not failures else "FAIL",
          "quality_verdict": "PASS" if not failures else "FAIL", "failures": failures,
          "audit_originals": 256, "audit_formats_are_correlated_views": True,
          "formats": trained, "untrained": untrained, "dataset_gate": gate,
          "candidate": frozen["candidate"], "restoration": restored,
          "audit_sha256": frozen["audit_sha256"], "parser": "not attempted",
          "runtime": "not attempted", "node_qualification": "not attempted",
          "release_gate": "INCOMPLETE", "scope": CONTRACT["release_status"]})
    if failures:
        raise RuntimeError("frozen audit quality gates failed; no adaptation allowed")


if __name__ == "__main__":
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("initial", "select", "final"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    {"initial": initial, "select": select, "final": final}[args.stage](args.root)
