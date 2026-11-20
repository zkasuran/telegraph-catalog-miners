# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Reciprocal literal source search. Annotation witnesses never imply preferences."""
import argparse
import itertools
import json
import re
import time
from pathlib import Path

from source import contains, norm, sha, write

MAX_ROWS = 2000
MAX_VOCABULARY = 20000
MAX_VOCABULARY_BYTES = 2 << 20
MAX_NODES = 250000
MAX_PROPOSALS = 20000
MAX_DATA_BYTES = 32 << 20
WORD = re.compile(r"\w")


def compile_vocabulary(vocabulary):
    root, nodes = {}, 1
    for phrase in sorted(vocabulary):
        node = root
        for character in phrase:
            if character not in node:
                node[character] = {}
                nodes += 1
                if nodes > MAX_NODES:
                    raise ValueError("literal trie ceiling exceeded")
            node = node[character]
        node[None] = phrase
    return root


def literal_presence(passage, vocabulary, compiled=None):
    """All overlapping boundary-aware phrases, including shorter nested phrases."""
    root = compile_vocabulary(vocabulary) if compiled is None else compiled
    value, found = norm(passage), set()
    for start in range(len(value)):
        if start and WORD.fullmatch(value[start - 1]):
            continue
        node = root
        for end in range(start, len(value)):
            node = node.get(value[end])
            if node is None:
                break
            if None in node and (end + 1 == len(value) or not WORD.fullmatch(value[end + 1])):
                found.add(node[None])
    return found


def expand(rows, prior):
    if not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("intact source record ceiling exceeded")
    vocabulary = {norm(k) for r in rows for k in r["literal_keywords"] if 4 <= len(norm(k)) <= 96}
    if len(vocabulary) > MAX_VOCABULARY or sum(len(k.encode()) for k in vocabulary) > MAX_VOCABULARY_BYTES:
        raise ValueError("source vocabulary ceiling exceeded")
    annotations = [{norm(k): k for k in r["literal_keywords"] if norm(k) in vocabulary} for r in rows]
    compiled = compile_vocabulary(vocabulary)
    presence = [literal_presence(r["text"], vocabulary, compiled) for r in rows]
    seen = {p["proposal_id"] for p in prior}
    if len(seen) != len(prior):
        raise ValueError("duplicate existing proposal ID")
    new, examined = [], 0
    for i, j in itertools.combinations(range(len(rows)), 2):
        examined += 1
        a, b = rows[i], rows[j]
        if a["group_id"] == b["group_id"] or not set(a["topics"]).intersection(b["topics"]):
            continue
        left = sorted(set(annotations[i]) & presence[j])
        right = sorted(set(annotations[j]) & presence[i])
        for first, other in itertools.product(left, right):
            if first == other:
                continue
            key = tuple(sorted((a["doi"], b["doi"]))) + tuple(sorted((first, other)))
            identity = sha("|".join(key).encode())
            if identity in seen:
                continue
            for term in (first, other):
                if not contains(a["text"], term) or not contains(b["text"], term):
                    raise ValueError("trie/primary literal boundary disagreement")
            seen.add(identity)
            new.append({"proposal_id": identity, "doi_a": a["doi"], "doi_b": b["doi"],
                        "term_a": annotations[i][first], "term_b": annotations[j][other],
                        "status": "unreviewed", "literal_in_both": True, "off_topic_donor": False,
                        "construction": "Separate original annotation witnesses; neither annotation membership nor lexical presence infers centrality.",
                        "annotation_witness_a": {"doi": a["doi"], "term": annotations[i][first]},
                        "annotation_witness_b": {"doi": b["doi"], "term": annotations[j][other]}})
            if len(prior) + len(new) > MAX_PROPOSALS:
                raise ValueError("expanded proposal ceiling exceeded, no partial corpus promoted")
    return prior + sorted(new, key=lambda p: p["proposal_id"]), {
        "intact_records_examined": len(rows), "source_pairs_examined": examined,
        "original_proposals_preserved": len(prior), "additional_unreviewed_proposals": len(new),
        "source_annotation_vocabulary": len(vocabulary),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--reviews", type=Path, help="prior explicit reviews to preserve; defaults to the source directory's parent reviews.json")
    args = parser.parse_args()
    started = time.monotonic()
    paths = [args.data / name for name in ("intact-articles.jsonl", "proposals.jsonl")]
    if any(p.stat().st_size > MAX_DATA_BYTES for p in paths):
        raise ValueError("bounded source-data input exceeded")
    original_bytes = [p.read_bytes() for p in paths]
    rows, prior = ([json.loads(line) for line in raw.decode().splitlines()] for raw in original_bytes)
    proposals, report = expand(rows, prior)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "proposals.jsonl"
    raw = "".join(json.dumps(p, ensure_ascii=False, allow_nan=False) + "\n" for p in proposals).encode()
    if len(raw) > MAX_DATA_BYTES:
        raise ValueError("expanded proposal byte ceiling exceeded")
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("refusing to replace an existing expansion snapshot")
    path.write_bytes(raw)
    intact_path = args.out / "intact-articles.jsonl"
    if intact_path.exists() and intact_path.read_bytes() != original_bytes[0]:
        raise ValueError("refusing to replace existing intact-source snapshot")
    intact_path.write_bytes(original_bytes[0])
    review_path = args.out / "reviews.json"
    if not review_path.exists():
        seed = args.reviews or args.data.parent / "reviews.json"
        if seed.stat().st_size > 1 << 20:
            raise ValueError("prior review byte ceiling exceeded")
        review_path.write_bytes(seed.read_bytes())
    write(args.out / "expansion-report.json", {"schema_version": 1, "family": "KEYWORD_EXTRACTION", **report,
          "input_sha256": {p.name: sha(raw) for p, raw in zip(paths, original_bytes, strict=True)},
          "output_sha256": sha(raw), "model_operations": 0, "elapsed_seconds": time.monotonic() - started,
          "scope": "Every intact record in the retained bounded sample, not all 19699 v1 source files. Proposals remain unreviewed."})
    print(json.dumps(report))


if __name__ == "__main__":
    main()
