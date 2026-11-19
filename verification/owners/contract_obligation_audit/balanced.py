# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Balanced development and a fresh source/entity-disjoint reserved audit."""
import argparse
import json
from collections import Counter
from pathlib import Path

from contracts import HERE, LABELS, check_config, sha, strict_json, write_json
from data import check_splits, collect_source, download, pair


def allocate(pool, hypotheses, count, group_cap, preferred=(), quotas=None):
    selected, used, groups, allocated = [], set(), Counter(), Counter()
    preferred = set(preferred)
    buckets = {}
    for hypothesis in hypotheses:
        for label in LABELS:
            buckets[hypothesis, label] = sorted((r for r in pool if r["hypothesis_id"] == hypothesis and r["label"] == label),
                key=lambda r: (r["source_group"] not in preferred,
                               sha(("balanced-v2:"+r["document_id"]+":"+hypothesis).encode())))
    while len(selected) < count:
        changed = False
        for hypothesis in hypotheses:
            if quotas and allocated[hypothesis] >= quotas[hypothesis]:
                continue
            # Select a balanced pair of distinct original evidence records.
            positive = negative = None
            for rejected in buckets[hypothesis, "Contradiction"]:
                if rejected["normalized_evidence_sha256"] in used or groups[rejected["source_group"]] >= group_cap:
                    continue
                for candidate in buckets[hypothesis, "Entailment"]:
                    if (candidate["normalized_evidence_sha256"] not in used
                            and candidate["normalized_evidence_sha256"] != rejected["normalized_evidence_sha256"]
                            and groups[candidate["source_group"]] + int(candidate["source_group"] == rejected["source_group"]) < group_cap):
                        positive, negative = candidate, rejected
                        break
                if positive is not None:
                    break
            if positive is None:
                continue
            for row in (negative, positive):
                selected.append(row)
                used.add(row["normalized_evidence_sha256"])
                groups[row["source_group"]] += 1
            allocated[hypothesis] += 1
            changed = True
            if len(selected) == count:
                break
        if not changed:
            raise ValueError(f"not enough balanced clean source pairs: {len(selected)}/{count}; {dict(allocated)}")
    if set(allocated) != set(hypotheses):
        raise ValueError("every claimed hypothesis requires both source labels")
    return selected


def construct(archive, config, policy):
    eligible, source = collect_source(archive, config)
    forbidden = set(policy["retired_groups"])
    audit_forbidden = set(policy["audit_forbidden_groups"])
    prior_evidence = set(policy["audit_forbidden_evidence"])
    hypotheses = policy["hypotheses"]
    clean = {split: [r for r in rows if r["source_group"] not in forbidden] for split, rows in eligible.items()}
    allowed = set(policy["allowed_source_records"])
    clean = {split: [row for row in rows if row["document_id"]+":"+row["hypothesis_id"]+":"+row["evidence_sha256"] in allowed]
             for split, rows in clean.items()}
    train = allocate(clean["train"], hypotheses, config["limits"]["train"], 4, policy["prior_training_groups"],
                     quotas=policy["training_pairs_per_hypothesis"])
    # Development coverage quotas were fixed from train/development inventory,
    # not from an audit result or a model score.
    validation = allocate(clean["validation"], hypotheses, config["limits"]["validation"], 4,
                          quotas=policy["validation_pairs_per_hypothesis"])
    development_groups = {r["source_group"] for r in train + validation}
    development_evidence = {r["normalized_evidence_sha256"] for r in train + validation}
    audit_pool = [r for rows in clean.values() for r in rows
                  if r["source_group"] not in audit_forbidden | development_groups
                  and r["normalized_evidence_sha256"] not in prior_evidence | development_evidence]
    # A group is never moved into train after reserving the independent audit.
    # No audit metric is calculated here or used to select the candidate.
    audit = allocate(audit_pool, hypotheses, config["limits"]["audit"], 2)
    splits = {}
    for split, rows in (("train", train), ("validation", validation), ("audit", audit)):
        splits[split] = [pair({**r, "original_source_split": r["split"], "split": split}) for r in rows]
        counts = Counter((r["hypothesis_id"], r["label"]) for r in rows)
        if any(counts[h, "Entailment"] != counts[h, "Contradiction"] for h in hypotheses):
            raise ValueError("hypothesis-conditioned source labels are not balanced")
    check_splits(splits, config)
    if any(r["source_group"] in forbidden for rows in splits.values() for r in rows):
        raise ValueError("retired audit group entered a new split")
    if any(r["source_group"] in audit_forbidden for r in splits["audit"]):
        raise ValueError("development-exposed group entered a new audit")
    return splits, {**source, "generation": 3, "selection_basis": "train/development coverage; audit source sampling only, no audit model or shortcut scores",
                    "hypotheses": hypotheses, "retired_groups_excluded": len(forbidden),
                    "audit_forbidden_groups": sorted(audit_forbidden), "policy_sha256": sha((HERE / "source-policy.json").read_bytes()),
                    "independent_groups": len({r["source_group"] for r in splits["audit"]})}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    config = check_config(strict_json((HERE / "config.json").read_bytes()))
    policy = strict_json((HERE / "source-policy.json").read_bytes())
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / (name + ".jsonl")).exists() for name in config["limits"]):
        raise ValueError("do not overwrite a prior source generation")
    archive = args.archive or args.out / "source.zip"
    if not args.archive:
        download(config["archive_url"], archive, config["archive_sha256"])
    if sha(archive.read_bytes()) != config["archive_sha256"]:
        raise ValueError("source archive identity mismatch")
    splits, source = construct(archive, config, policy)
    files = {}
    for name, rows in splits.items():
        raw = b"".join(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()+b"\n" for row in rows)
        (args.out / (name+".jsonl")).write_bytes(raw)
        files[name] = {"sha256": sha(raw), "pairs": len(rows), "groups": len({r["source_group"] for r in rows})}
    write_json(args.out / "source-manifest.json", {**source, "files": files,
               "archive_sha256": config["archive_sha256"],
               "attribution": "Koreeda and Manning (2021), ContractNLI, CC BY 4.0",
               "modifications": "Complete annotated evidence, balanced source selection and AI-assisted claim-status templates"})
    print(json.dumps({"files": files, "hypotheses": source["hypotheses"], "audit_metrics": "not inspected",
                      "retired_groups_excluded": len(policy["retired_groups"])}, indent=2))


if __name__ == "__main__":
    main()
