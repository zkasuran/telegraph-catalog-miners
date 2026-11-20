# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Render original source text for centrality review, without any model scoring."""
import argparse
import json
from pathlib import Path

from source import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--indices", help="comma-separated indices in the current frozen proposal file")
    parser.add_argument("--packed", action="store_true", help="view resource-disjoint unreviewed proposals, not approved data")
    parser.add_argument("--split", choices=("train", "validation", "test"))
    args = parser.parse_args()
    articles = {r["doi"]: r for r in (json.loads(x) for x in (args.data / "intact-articles.jsonl").read_text().split("\n") if x)}
    proposals = [json.loads(x) for x in (args.data / "proposals.jsonl").read_text().split("\n") if x]
    selected = [int(x) for x in args.indices.split(",")] if args.indices else list(range(args.offset, min(len(proposals), args.offset + args.count)))
    if args.packed:
        report = json.loads((args.data / "proposal-matching.json").read_text())
        for name, digest in report["input_sha256"].items():
            if sha((args.data / name).read_bytes()) != digest:
                raise ValueError("stale proposal packing, rederive from the current review state")
        packed = [p for p in report["selected"] if not args.split or p["prospective_split"] == args.split]
        selected = [p["index"] for p in packed[args.offset:args.offset + args.count]]
        for p in packed:
            if proposals[p["index"]]["proposal_id"] != p["proposal_id"]:
                raise ValueError("packed proposal identity mismatch")
    for i in selected:
        pair = proposals[i]
        print(f"\nPAIR {i} {pair['proposal_id']}\nTERMS {pair['term_a']} | {pair['term_b']}")
        for side in ("a", "b"):
            row = articles[pair["doi_" + side]]
            if args.summary:
                print(side.upper(), row["doi"], row["title"], "TOPICS", row["topics"])
                continue
            print(side.upper(), row["doi"], "TOPICS", row["topics"], "\n", row["text"], "\nAUTHOR KEYWORDS", row["author_keywords"])


if __name__ == "__main__":
    main()
