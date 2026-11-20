# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Generation-two source preparation, with a freeze before replacement audit."""
import argparse
import json
from pathlib import Path

import source_data

HERE = Path(__file__).resolve().parent
digest = source_data.digest
norm = source_data.norm
render = source_data.render
write_json = source_data.write_json
read_splits = source_data.read_splits
COUNTS = {"train": 768, "validation": 128, "audit": 64}
RECIPE = {"max_length": 256, "lora_rank": 8, "lora_alpha": 32, "batch_size": 1,
          "grad_accum": 16, "epochs": 3, "eval_batch": 1, "seed": 42}


def payload(rows):
    return "".join(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n" for row in rows).encode()


def source_pool(out, cache=None):
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    pins = source_data.get_sources(out / "sources", cache)
    tokenizer = Tokenizer.from_file(str(out / "sources/tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    columns = ["message_id", "parent_id", "message_tree_id", "text", "role", "lang", "rank",
               "review_count", "review_result", "deleted", "synthetic", "labels"]
    rows = {}
    for split in ("train", "validation"):
        table = pq.read_table(out / "sources" / ("oasst-" + split + ".parquet"), columns=columns)
        if table.num_rows > pins["limits"]["max_messages"]:
            raise ValueError("source inventory exceeds its ceiling")
        rows[split] = table.to_pylist()
    pools, inventory = source_data.preferences(rows, lambda p, a: tokenizer.encode(render(p, a)).ids)
    exclusion_raw = (HERE / "exposed-exclusions.json").read_bytes()
    exclusions = json.loads(exclusion_raw)
    groups = set(exclusions["group_ids"])
    fields = set(exclusions["normalized_field_sha256"])
    available = []
    for values in pools.values():
        for row in values:
            if row["source_split"] != "train" or row["group_id"] in groups:
                continue
            if any(digest(norm(row[key]).encode()) in fields for key in ("prompt", "chosen", "rejected")):
                continue
            available.append(row)
    return available, tokenizer, pins, inventory, digest(exclusion_raw)


def choose(rows, count, salt):
    if count % 2:
        raise ValueError("source length-balance count must be even")
    selected = []
    for longer in (True, False):
        candidates = [row for row in rows if (len(row["chosen"]) > len(row["rejected"])) == longer
                      and len(row["chosen"]) != len(row["rejected"])]
        candidates.sort(key=lambda row: digest((salt + row["group_id"]).encode()))
        if len(candidates) < count // 2:
            raise ValueError("insufficient independent source components")
        selected.extend(candidates[:count // 2])
    return sorted(selected, key=lambda row: row["group_id"])


def prepare_development(out, cache=None, kit=None):
    rows, tokenizer, pins, inventory, exclusion_sha = source_pool(out, cache)
    train = choose(rows, COUNTS["train"], "qa-v2-train-20261120")
    remaining = [row for row in rows if row["group_id"] not in {r["group_id"] for r in train}]
    validation = choose(remaining, COUNTS["validation"], "qa-v2-dev-20261120")
    source_data.check_splits({"train": train, "validation": validation})
    for name, selected in (("train", train), ("validation", validation)):
        (out / (name + ".jsonl")).write_bytes(payload(selected))
    design = {"schema_version": 1, "generation": 2, "source_pins": pins, "recipe": RECIPE,
              "files": {name: {"rows": len(selected), "groups": len(selected), "sha256": digest(payload(selected))}
                        for name, selected in (("train", train), ("validation", validation))},
              "exposed_exclusion_sha256": exclusion_sha, "inventory": inventory,
              "source_components_available": len(rows), "audit_constructed": False,
              "scope": pins["scope"], "model_operations": False}
    write_json(out / "development-design.json", design)
    # Train-fitted source controls only. No audit pairs exist at this phase.
    from controls import fit_and_gate_development
    kit = kit or out.parent / ("telegraph-wasm-baseline-" + pins["kit_revision"])
    report, coefficients = fit_and_gate_development(train, validation, tokenizer, kit)
    write_json(out / "development-data-gate.json", report)
    if not report["ok"]:
        raise ValueError("generation-two training/development source gate failed")
    bundle = HERE / "baselines/frozen-controls.json"
    if bundle.exists():
        (out / "frozen-controls.json").write_bytes(bundle.read_bytes())
    else:
        write_json(out / "frozen-controls.json", coefficients)
    freeze = {"schema_version": 1, "generation": 2, "design_sha256": digest((out / "development-design.json").read_bytes()),
              "frozen_controls_sha256": digest((out / "frozen-controls.json").read_bytes()),
              "training_files": design["files"], "recipe": RECIPE, "audit_used_for_selection": False}
    write_json(out / "development-freeze.json", freeze)
    return design, freeze


def prepare_audit(out, cache=None, kit=None):
    # This phase is refused unless the exact development design already froze.
    freeze = json.loads((out / "development-freeze.json").read_text())
    design_raw = (out / "development-design.json").read_bytes()
    if freeze["design_sha256"] != digest(design_raw) or freeze["recipe"] != RECIPE:
        raise ValueError("development freeze identity mismatch")
    design = json.loads(design_raw)
    for name, pin in freeze["training_files"].items():
        if digest((out / (name + ".jsonl")).read_bytes()) != pin["sha256"]:
            raise ValueError("development data changed after freeze")
    if digest((out / "frozen-controls.json").read_bytes()) != freeze["frozen_controls_sha256"]:
        raise ValueError("train-only coefficients changed after freeze")
    rows, _, pins, _, exclusion_sha = source_pool(out, cache)
    if exclusion_sha != design["exposed_exclusion_sha256"]:
        raise ValueError("exposure exclusion changed after freeze")
    prior = {name: [json.loads(line) for line in (out / (name + ".jsonl")).read_text().split("\n") if line]
             for name in ("train", "validation")}
    used = {row["group_id"] for values in prior.values() for row in values}
    audit = choose([row for row in rows if row["group_id"] not in used], COUNTS["audit"], "qa-v2-audit-after-freeze-20261120")
    source_data.check_splits({**prior, "audit": audit})
    (out / "audit.jsonl").write_bytes(payload(audit))
    report = {"status": "SOURCE_ONLY", "generation": 2, "training_allowed": True, "scope": pins["scope"],
              "files": {**design["files"], "audit": {"rows": 64, "groups": 64, "sha256": digest(payload(audit))}},
              "source_pins": pins, "development_freeze_sha256": digest((out / "development-freeze.json").read_bytes()),
              "exposed_exclusion_sha256": exclusion_sha, "normalized_overlap": 0,
              "audit_created_after_design_freeze": True, "audit_used_for_selection": False,
              "original_preferences_unchanged": True, "source_grants_retained": "Apache-2.0",
              "old_audit_status": "exposed failed diagnostic evidence, excluded"}
    write_json(out / "data-report.json", report)
    (out / "LICENSE").write_bytes((HERE / "UPSTREAM-APACHE-2.0.txt").read_bytes())
    (out / "NOTICE").write_bytes((HERE / "NOTICE").read_bytes())
    return report


def prepare(out, cache=None, kit=None):
    prepare_development(out, cache, kit)
    return prepare_audit(out, cache, kit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("development", "audit", "all"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--kit", type=Path)
    args = parser.parse_args()
    action = {"development": prepare_development, "audit": prepare_audit, "all": prepare}[args.phase]
    print(json.dumps(action(args.out, args.source_cache, args.kit), indent=2))
