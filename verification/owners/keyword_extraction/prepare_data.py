# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Intact source grouping and explicit review proposals. Never infers gold labels."""
import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from scorer import render
from source import contains, norm, sha, write

TOKENIZER_SHA = "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456"
MAX_TOKENS = 384


class Union:
    def __init__(self, size):
        self.parent = list(range(size))

    def root(self, node):
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        self.parent[max(a, b)] = min(a, b)


def group_sources(rows):
    union, owners = Union(len(rows)), {}
    for i, row in enumerate(rows):
        keys = ["doi:" + norm(row["doi"]), "text:" + norm(row["text"]), "abstract:" + norm(row["abstract"])]
        keys += ["author:" + norm(a) for a in row["authors"]]
        for key in keys:
            if key in owners:
                union.join(i, owners[key])
            owners[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[union.root(i)].append(row)
    for members in groups.values():
        identity = sha("\n".join(sorted(norm(r["doi"]) for r in members)).encode())
        topics = sorted({t for r in members for t in r["topics"]})
        for row in members:
            row["group_id"] = identity
            row["component_topics"] = topics
    return groups


def prompt(row):
    return "Return the single most central keyphrase for this research abstract.\n\n" + row["text"]


def whole_annotation(row):
    """One literal, complete original-annotation wire rendering for every gate."""
    return json.dumps({"keywords": row["author_keywords"]}, ensure_ascii=False, separators=(",", ":"))


def answer(term, style):
    if style == "plain":
        return term
    if style == "array":
        return json.dumps([term], ensure_ascii=False, separators=(",", ":"))
    if style == "object":
        return json.dumps({"keywords": [term]}, ensure_ascii=False, separators=(",", ":"))
    raise ValueError("unknown response style")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    if sha(args.tokenizer.read_bytes()) != TOKENIZER_SHA:
        raise ValueError("authoritative tokenizer hash mismatch")
    tok = Tokenizer.from_file(str(args.tokenizer))
    tok.no_truncation()
    tok.no_padding()
    rows = [json.loads(line) for line in args.articles.read_text().split("\n") if line]
    args.out.mkdir(parents=True, exist_ok=True)
    groups = group_sources(rows)
    fit, counts = [], Counter()
    for row in rows:
        whole = whole_annotation(row)
        row["whole_annotation_render_tokens"] = len(tok.encode(render(prompt(row), whole)).ids)
        row["prompt_bytes"] = len(prompt(row).encode())
        if row["whole_annotation_render_tokens"] > MAX_TOKENS:
            counts["whole_record_over_384_tokens"] += 1
        else:
            fit.append(row)
    # Two scientifically different documents sharing both literal keyword
    # candidates permit exact reciprocal answer reuse. No answer-only tell and
    # no unequal lexical-presence tell. These are proposals pending centrality
    # review, NOT preferences inferred merely from annotation membership.
    index = defaultdict(list)
    for row in fit:
        for key in {norm(k) for k in row["literal_keywords"] if len(norm(k)) >= 4}:
            index[key].append(row)
    proposed, seen = [], set()
    for term, members in sorted(index.items()):
        for a, b in itertools.combinations(members, 2):
            if a["group_id"] == b["group_id"]:
                continue
            if not set(a["topics"]).intersection(b["topics"]):
                continue
            alternatives = [k for k in a["literal_keywords"] if norm(k) != term and contains(b["text"], k)]
            alternatives += [k for k in b["literal_keywords"] if norm(k) != term and contains(a["text"], k)]
            for other in alternatives:
                key = tuple(sorted((a["doi"], b["doi"]))) + tuple(sorted((term, norm(other))))
                if key in seen:
                    continue
                seen.add(key)
                first = next(k for k in a["literal_keywords"] if norm(k) == term)
                proposed.append({"proposal_id": sha("|".join(key).encode()), "doi_a": a["doi"], "doi_b": b["doi"],
                                 "term_a": first, "term_b": other, "status": "unreviewed",
                                 "literal_in_both": True, "off_topic_donor": False})
    for name, items in (("intact-articles.jsonl", fit), ("proposals.jsonl", proposed)):
        (args.out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items))
    write(args.out / "fit-report.json", {"originals": len(rows), "author_doi_text_components": len(groups),
          "component_sizes": sorted((len(v) for v in groups.values()), reverse=True),
          "whole_annotations_fit_384": len(fit), "rejected": dict(counts), "reciprocal_proposals": len(proposed),
          "tokenizer_sha256": TOKENIZER_SHA, "max_tokens": MAX_TOKENS,
          "request_256_bytes_fit": sum(r["prompt_bytes"] <= 256 for r in fit),
          "minimum_render_tokens": min(r["whole_annotation_render_tokens"] for r in fit),
          "maximum_render_tokens": max(r["whole_annotation_render_tokens"] for r in fit),
          "files": {n: sha((args.out / n).read_bytes()) for n in ("intact-articles.jsonl", "proposals.jsonl")},
          "limits": "Proposals are not approved pairs. Shared normalized authors are conservative identity proxies, not resolved author IDs. Complete source grouping precedes token filtering."})
    print(json.dumps({"intact_articles": len(fit), "source_components": len(groups), "proposals": len(proposed)}))


if __name__ == "__main__":
    main()
