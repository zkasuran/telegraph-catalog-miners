# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Assemble only reviewed centrality preferences. Never promote proposals by default."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from gates import check_splits
from prepare_data import (
    MAX_TOKENS,
    TOKENIZER_SHA,
    Union,
    answer,
    prompt,
    whole_annotation,
)
from scorer import render
from source import contains, norm, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    review = json.loads(args.reviews.read_text())
    approved = [r for r in review["reviews"] if r["status"] == "approve"]
    args.out.mkdir(parents=True, exist_ok=True)
    if not approved:
        write(args.out / "assembly-failure.json", {"verdict": "FAIL", "reason": "No approved centrality preferences. Original annotations and proposals cannot supply preferences by themselves.",
              "approved": 0, "review_sha256": sha(args.reviews.read_bytes()), "model_operations": 0})
        raise ValueError("no approved centrality preferences")
    rows = [json.loads(x) for x in (args.data / "intact-articles.jsonl").read_text().split("\n") if x]
    proposals = {r["proposal_id"]: r for r in (json.loads(x) for x in (args.data / "proposals.jsonl").read_text().split("\n") if x)}
    ids = {r["doi"]: i for i, r in enumerate(rows)}
    union, owners = Union(len(rows)), {}
    for i, row in enumerate(rows):
        if row["group_id"] in owners:
            union.join(i, owners[row["group_id"]])
        owners[row["group_id"]] = i
    answer_owners = {}
    for r in approved:
        p = proposals[r["proposal_id"]]
        if r.get("choice_a") not in {"term_a", "term_b"} or r.get("choice_b") not in {"term_a", "term_b"} or r["choice_a"] == r["choice_b"]:
            raise ValueError("review must explicitly support opposite centrality preferences")
        if r.get("review_origin") != "AI-assisted" or not r.get("reason_a") or not r.get("reason_b"):
            raise ValueError("explicit AI-assisted review and reasons required, no human-review substitution")
        union.join(ids[p["doi_a"]], ids[p["doi_b"]])
        for term in (p["term_a"], p["term_b"]):
            key = norm(answer(term, "array"))
            if key in answer_owners:
                union.join(ids[p["doi_a"]], answer_owners[key])
            answer_owners[key] = ids[p["doi_a"]]
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[union.root(i)].append(row)
    group, split = {}, {}
    for root, members in components.items():
        g = sha("\n".join(sorted(r["doi"] for r in members)).encode())
        group[root] = g
        # Whole donor-author components touching this held-out topic stay test.
        split[root] = "test" if any("Neuroscience" in r["component_topics"] for r in members) else "validation" if int(g[:8], 16) % 5 == 0 else "train"
    from tokenizers import Tokenizer
    if sha(args.tokenizer.read_bytes()) != TOKENIZER_SHA:
        raise ValueError("tokenizer identity mismatch")
    tok = Tokenizer.from_file(str(args.tokenizer))
    tok.no_truncation()
    tok.no_padding()
    splits = {s: [] for s in ("train", "validation", "test")}
    used = set()
    for r in approved:
        p = proposals[r["proposal_id"]]
        for side in ("a", "b"):
            row = rows[ids[p["doi_" + side]]]
            if row["doi"] in used:
                raise ValueError("one reviewed preference per DOI required")
            used.add(row["doi"])
            chosen = p[r["choice_" + side]]
            rejected = p["term_b" if r["choice_" + side] == "term_a" else "term_a"]
            if not contains(row["text"], chosen) or not contains(row["text"], rejected):
                raise ValueError("both contrasted terms must be literal in this exact passage")
            # Check whole original annotations, not just shorter target answers.
            whole = whole_annotation(row)
            if len(tok.encode(render(prompt(row), whole)).ids) > MAX_TOKENS:
                raise ValueError("whole original annotation no longer fits")
            for style in ("plain", "array", "object"):
                for value in (chosen, rejected):
                    if len(tok.encode(render(prompt(row), answer(value, style))).ids) > MAX_TOKENS:
                        raise ValueError("complete preference format exceeds token cap")
            root = union.root(ids[row["doi"]])
            splits[split[root]].append({"prompt": prompt(row), "chosen": answer(chosen, "array"),
                "rejected": answer(rejected, "array"), "chosen_term": chosen, "rejected_term": rejected,
                "group_id": group[root], "source": row, "label_kind": "preference_pair",
                "review_origin": "AI-assisted", "centrality_reason": r["reason_" + side],
                "off_topic_donor": False, "proposal_id": p["proposal_id"]})
    (args.out / "reviewed-pairs.jsonl").write_text("".join(json.dumps({**r, "provisional_split": s}, ensure_ascii=False) + "\n" for s, pairs in splits.items() for r in pairs))
    write(args.out / "reviewed-manifest.json", {"family": "KEYWORD_EXTRACTION", "max_tokens": MAX_TOKENS,
          "reviews_sha256": sha(args.reviews.read_bytes()), "reviewed_pairs_sha256": sha((args.out / "reviewed-pairs.jsonl").read_bytes()),
          "by_split": {s: {"pairs": len(pairs), "groups": len({r["group_id"] for r in pairs})} for s, pairs in splits.items()},
          "status": "Reviewed source-derived candidates only. Not approved training data until every gate passes.",
          "preference_origin": "AI-assisted comparative centrality/informativeness review. No human review claimed."})
    try:
        check_splits(splits)
    except ValueError as exc:
        write(args.out / "assembly-failure.json", {"verdict": "FAIL", "reason": str(exc), "model_operations": 0,
              "review_sha256": sha(args.reviews.read_bytes())})
        raise
    minimums = {"train": 64, "validation": 16, "test": 64}
    if any(len(splits[s]) < minimums[s] for s in splits):
        write(args.out / "assembly-failure.json", {"verdict": "FAIL", "reason": "insufficient reviewed independent pairs for the 64/16/64 pilot", "model_operations": 0,
              "observed_pairs": {s: len(splits[s]) for s in splits}, "review_sha256": sha(args.reviews.read_bytes())})
        raise ValueError("insufficient independent reviewed pairs for the 64/16/64 pilot")
    if any(len({r["group_id"] for r in splits[s]}) < minimums[s] // 2 for s in splits):
        raise ValueError("too few independent source components for the pilot")
    files = {}
    for s, pairs in splits.items():
        path = args.out / (s + ".jsonl")
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pairs))
        files[s] = {"sha256": sha(path.read_bytes()), "pairs": len(pairs), "groups": len({r["group_id"] for r in pairs})}
    write(args.out / "approved-manifest.json", {"family": "KEYWORD_EXTRACTION", "max_tokens": MAX_TOKENS, "files": files,
          "held_out_group_rule": "topic-held-out plus DOI-author-donor components", "held_out_topic": "Neuroscience",
          "review_sha256": sha(args.reviews.read_bytes()), "articles_sha256": sha((args.data / "intact-articles.jsonl").read_bytes()),
          "sources": [r["source"] for pairs in splits.values() for r in pairs], "normalized_cross_split_overlap": 0,
          "preference_origin": "AI-assisted constructed centrality preferences. Not human-reviewed."})
    inventory = args.data.parent / "sources" / "inventory.json"
    (args.out / "inventory.json").write_bytes(inventory.read_bytes())


if __name__ == "__main__":
    main()
