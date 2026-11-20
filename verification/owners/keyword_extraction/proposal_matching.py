# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pack unreviewed proposals by source/answer resources. Never produces preferences."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from assemble import validate_reviews
from expand_proposals import MAX_DATA_BYTES, MAX_PROPOSALS, MAX_ROWS
from source import contains, norm, sha, write


def matching(rows, proposals, reviews, priority_split=None):
    indexed = {p["proposal_id"]: p for p in proposals}
    approved = validate_reviews(reviews, indexed)
    by_doi = {r["doi"]: r for r in rows}
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group_id"]].append(r)
    reviewed = {r["proposal_id"] for r in reviews["reviews"]}
    blocked = set()
    def resources(p):
        return {"source:" + by_doi[p[k]]["group_id"] for k in ("doi_a", "doi_b")} | {"answer:" + norm(p[k]) for k in ("term_a", "term_b")}
    for r in approved:
        blocked.update(resources(indexed[r["proposal_id"]]))
    candidates = [(i, p, resources(p)) for i, p in enumerate(proposals) if p["proposal_id"] not in reviewed]
    candidates = [(i, p, keys) for i, p, keys in candidates if not keys & blocked]
    degree = Counter(key for _, _, keys in candidates for key in keys)
    selected = []
    def prospective(p):
        members = [r for key in ("doi_a", "doi_b") for r in by_group[by_doi[p[key]]["group_id"]]]
        group = sha("\n".join(sorted(r["doi"] for r in members)).encode())
        split = "test" if any("Neuroscience" in r["component_topics"] for r in members) else "validation" if int(group[:8], 16) % 5 == 0 else "train"
        return group, split
    projected = {p["proposal_id"]: prospective(p) for _, p, _ in candidates}
    # Nested wording is often the same concept, so inspect distinct phrases first.
    # This orders review work only. It neither labels nor approves any contrast.
    def priority(value):
        _, p, keys = value
        nested = contains(p["term_a"], p["term_b"]) or contains(p["term_b"], p["term_a"])
        preferred_split = priority_split is not None and projected[p["proposal_id"]][1] != priority_split
        return preferred_split, nested, sum(degree[k] for k in keys), p["proposal_id"]
    for i, p, keys in sorted(candidates, key=priority):
        if keys & blocked:
            continue
        group, split = projected[p["proposal_id"]]
        blocked.update(keys)
        selected.append({"index": i, "proposal_id": p["proposal_id"], "doi_a": p["doi_a"], "doi_b": p["doi_b"],
                         "term_a": p["term_a"], "term_b": p["term_b"], "prospective_split": split,
                         "prospective_group_id": group, "status": "unreviewed"})
    return {"schema_version": 1, "family": "KEYWORD_EXTRACTION", "available_resource_disjoint_proposals": len(candidates),
            "available": [{"index": i, "proposal_id": p["proposal_id"], "prospective_split": projected[p["proposal_id"]][1], "status": "unreviewed"} for i, p, _ in candidates],
            "available_by_prospective_split": dict(Counter(s for _, s in projected.values())), "packing_priority": priority_split,
            "packed_unreviewed_contrasts": len(selected), "by_prospective_split": dict(Counter(p["prospective_split"] for p in selected)),
            "selected": selected, "model_operations": 0,
            "limits": "Greedy packing of unreviewed proposals, not a maximum matching or approved preferences. Prospective split assumes the existing grouping and no additional connections. Full assembly rechecks everything."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--priority-split", choices=("train", "validation", "test"))
    args = parser.parse_args()
    paths = {name: args.data / name for name in ("intact-articles.jsonl", "proposals.jsonl", "reviews.json")}
    if any(path.stat().st_size > MAX_DATA_BYTES for path in paths.values()):
        raise ValueError("proposal packing input byte ceiling exceeded")
    rows = [json.loads(line) for line in paths["intact-articles.jsonl"].read_text().splitlines()]
    proposals = [json.loads(line) for line in paths["proposals.jsonl"].read_text().splitlines()]
    if len(rows) > MAX_ROWS or len(proposals) > MAX_PROPOSALS:
        raise ValueError("proposal packing row/proposal ceiling exceeded")
    report = matching(rows, proposals, json.loads(paths["reviews.json"].read_text()), args.priority_split)
    report["input_sha256"] = {name: sha(path.read_bytes()) for name, path in paths.items()}
    write(args.data / "proposal-matching.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"selected", "available"}}))


if __name__ == "__main__":
    main()
