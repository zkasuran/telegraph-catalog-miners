# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""RELISH human relation labels, exact metadata bindings and title-only screens."""
import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from source import HERE, SELECT, sha, write

WORDS = re.compile(r"[a-z0-9']+")
FORMATS = ("title", "json", "prose")
MAX_QUERY_BYTES = 256
MAX_TEXT_BYTES = 4096


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def words(text):
    return WORDS.findall(norm(text))


def render(prompt, answer):
    return f"### Task:\n{prompt}\n\n### Answer:\n{answer}"


def formats(title, doi):
    if not isinstance(title, str) or not title.strip() or len(title.encode()) > MAX_TEXT_BYTES:
        raise ValueError("title is empty or exceeds byte ceiling")
    if doi is not None and (not isinstance(doi, str) or len(doi) > 512 or not doi.startswith("https://doi.org/10.")):
        raise ValueError("DOI metadata shape")
    # Counts are omitted. A single cited title does not imply a total result count.
    return {"title": title,
            "json": json.dumps({"answer": title, "paper_titles": [title], "dois": [doi] if doi else []}, separators=(",", ":"), ensure_ascii=False),
            "prose": "Related paper: " + title + (" (" + doi + ")" if doi else "")}


class Components:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            previous = item
            item = self.parent[item]
            self.parent[previous] = root
        return root

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        self.parent[max(a, b)] = min(a, b)


def entities(metadata):
    components, seen = Components(), {}
    for ident, record in sorted(metadata.items()):
        if set(record) != set(SELECT.split(",")):
            raise ValueError("metadata projection contains unapproved fields")
        pmid = str(record["ids"].get("pmid", "")).rstrip("/").rsplit("/", 1)[-1]
        if pmid != ident or record["id"] != record["ids"].get("openalex"):
            raise ValueError("PMID/OpenAlex metadata identity mismatch")
        components.find(ident)
        keys = ["pmid:" + ident, "openalex:" + record["id"]]
        if record["title"] and isinstance(record["title"], str):
            keys.append("title:" + norm(record["title"]))
        if record["doi"]:
            keys.append("doi:" + norm(record["doi"]).removeprefix("https://doi.org/"))
        for key in keys:
            if key in seen:
                components.union(ident, seen[key])
            else:
                seen[key] = ident
    return components


def identity_keys(row):
    return {"entity:" + i for i in row["entity_components"]} | {"title:" + norm(row[field]) for field in ("query_title", "chosen_title", "rejected_title")}


def check_splits(splits):
    seen = {}
    groups = Counter()
    for split, rows in splits.items():
        for row in rows:
            if row["human_pairwise_vote"] is not False or row["source_label"] != "relevant_vs_irrelevant":
                raise ValueError("constructed preferences must retain honest source labels")
            if len(set(row["entity_components"])) != 3:
                raise ValueError("seed and both targets must be different source entities")
            for key in identity_keys(row):
                if key in seen:
                    raise ValueError("source entity/title reused in different originals: " + key)
                seen[key] = split
            groups[split] += 1
    return {"normalized_overlap": 0, "independent_groups": dict(groups)}


def candidate_pool(metadata, pool, seeds):
    components = entities(metadata)
    candidates, excluded = [], Counter()
    for seed in seeds:
        if seed not in metadata or not metadata[seed]["title"]:
            excluded["missing_seed_title"] += 1
            continue
        query = metadata[seed]["title"]
        if len(query.encode()) > MAX_QUERY_BYTES:
            excluded["query_over_256_bytes"] += 1
            continue
        if len(words(query)) < 4:
            excluded["underspecified_seed_title"] += 1
            continue
        good = [r for r in pool[seed]["relevant"] if r["pmid"] in metadata and metadata[r["pmid"]]["title"]]
        bad = [r for r in pool[seed]["irrelevant"] if r["pmid"] in metadata and metadata[r["pmid"]]["title"]]
        options = []
        for chosen in good:
            for rejected in bad:
                ids = [seed, chosen["pmid"], rejected["pmid"]]
                roots = [components.find(i) for i in ids]
                if len(set(roots)) != 3:
                    continue
                a, b = metadata[ids[1]], metadata[ids[2]]
                if bool(a["doi"]) != bool(b["doi"]):
                    continue
                wt = set(words(query))
                wa, wb = words(a["title"]), words(b["title"])
                if not wa or not wb:
                    continue
                # Mine compact title-level hard contrasts before any split or audit.
                difficulty = (abs(len(set(wa) & wt) - len(set(wb) & wt)), abs(len(wa) - len(wb)), abs(len(a["title"]) - len(b["title"])), sha((seed + ":" + ids[1] + ":" + ids[2]).encode()))
                options.append((difficulty, chosen, rejected, roots))
        if not options:
            excluded["no_distinct_bound_contrast"] += 1
            continue
        # Keep alternatives for the unsplit semantic review, not one metric-selected audit.
        for _, chosen, rejected, roots in sorted(options)[:3]:
            a, b = metadata[chosen["pmid"]], metadata[rejected["pmid"]]
            try:
                af, bf = formats(a["title"], a["doi"]), formats(b["title"], b["doi"])
            except ValueError:
                excluded["metadata_format_failure"] += 1
                continue
            block_id = sha((seed + ":" + chosen["pmid"] + ":" + rejected["pmid"]).encode())
            candidates.append({"block_id": block_id, "seed_pmid": seed, "chosen_pmid": chosen["pmid"],
                               "rejected_pmid": rejected["pmid"], "entity_components": roots,
                               "query_title": query, "chosen_title": a["title"], "rejected_title": b["title"],
                               "chosen_doi": a["doi"], "rejected_doi": b["doi"], "prompt": query,
                               "formats": {f: {"chosen": af[f], "rejected": bf[f]} for f in FORMATS},
                               "source_label": "relevant_vs_irrelevant", "human_pairwise_vote": False,
                               "source_human_annotations": {"chosen": chosen["annotation_uids"], "rejected": rejected["annotation_uids"]},
                               "pair_construction": "code pairing of uncontested original human article-relevance labels",
                               "title_visibility_status": "UNREVIEWED", "source": "RELISH_v1+OpenAlex-CC0"})
    return candidates, {"source_seeds_considered": len(seeds), "unsplit_candidate_blocks": len(candidates), "excluded": dict(excluded),
                        "metadata_records": len(metadata), "metadata_entity_components": len({components.find(i) for i in metadata}),
                        "source_metadata_scope": "all captured bindings, not all OpenAlex", "audit_scored": False, "model_loaded": False}


def source_visible(row, tokenizer):
    for f in FORMATS:
        sequences = [tokenizer.encode(render(row["prompt"], row["formats"][f][side])).ids for side in ("chosen", "rejected")]
        if any(len(s) > 256 for s in sequences) or sequences[0] == sequences[1]:
            return False
    return True


def select_reviewed(rows, decisions, sizes=None):
    sizes = sizes or {"train": 128, "validation": 32, "audit": 64}
    if set(sizes) != {"train", "validation", "audit"} or sizes != {"train": 128, "validation": 32, "audit": 64}:
        raise ValueError("fixed original-group quotas cannot be weakened")
    index = {r["block_id"]: r for r in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate candidate block")
    approved = []
    for decision in decisions:
        ident = decision["block_id"]
        if ident not in index or decision["reviewer_kind"] != "AI-assisted" or not decision["reason"].strip():
            raise ValueError("review must bind an original and honest reviewer type")
        if decision["decision"] == "approve":
            approved.append(index[ident])
        elif decision["decision"] != "reject":
            raise ValueError("unresolved review is not approval")
    if len({r["block_id"] for r in approved}) != len(approved):
        raise ValueError("duplicate approval")
    splits, used, offset = {}, set(), 0
    originals = []
    for row in sorted(approved, key=lambda r: sha(("RELISH-freeze-v1:" + r["block_id"]).encode())):
        keys = identity_keys(row)
        if keys & used:
            continue
        used.update(keys)
        originals.append({**row, "title_visibility_status": "AI-assisted source review before splitting"})
    if len(originals) < sum(sizes.values()):
        raise ValueError("insufficient independently reviewed source originals: " + str(len(originals)))
    for split, count in sizes.items():
        splits[split] = originals[offset:offset + count]
        offset += count
    check_splits(splits)
    return splits


def freeze(decisions_path):
    raw = (HERE / "review-pool.json").read_bytes()
    pool = json.loads(raw)
    decisions = json.loads(decisions_path.read_bytes())
    if decisions["review_pool_sha256"] != sha(raw) or decisions["schema_version"] != 1:
        raise ValueError("semantic review refers to another source pool")
    if decisions.get("audit_used_for_selection") is not False:
        raise ValueError("source review must precede all audit selection")
    splits = select_reviewed(pool["rows"], decisions["decisions"])
    root = HERE / "frozen"
    root.mkdir(exist_ok=True)
    manifest = {"schema_version": 1, "family": "ACADEMIC_SEARCH", "source_manifest_sha256": pool["source_manifest_sha256"],
                "review_pool_sha256": sha(raw), "review_sha256": sha(decisions_path.read_bytes()),
                "audit_used_for_selection": False, "grants": "RELISH annotations and OpenAlex metadata CC0", "splits": {},
                "formats": list(FORMATS), **check_splits(splits)}
    for split, rows in splits.items():
        write(root / (split + ".json"), rows)
        manifest["splits"][split] = {"rows": len(rows), "sha256": sha((root / (split + ".json")).read_bytes())}
    write(root / "freeze.json", manifest)
    return manifest


def prepare(tokenizer_path=None):
    metadata = json.loads((HERE / "metadata.json").read_bytes())
    pool = json.loads((HERE / "annotation-pool.json").read_bytes())
    inventory = json.loads((HERE / "source-inventory.json").read_bytes())
    rows, report = candidate_pool(metadata, pool, inventory["selected_seeds"])
    if tokenizer_path:
        raw = tokenizer_path.read_bytes()
        if sha(raw) != "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456":
            raise ValueError("released tokenizer identity mismatch")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        visible = [r for r in rows if source_visible(r, tokenizer)]
        report.update({"token_visible_blocks": len(visible), "token_overflow_or_identity_exclusions": len(rows) - len(visible), "tokenizer_sha256": sha(raw)})
        rows = visible
    write(HERE / "review-pool.json", {"schema_version": 1, "source_manifest_sha256": sha((HERE / "source-inventory.json").read_bytes()), "rows": rows})
    write(HERE / "preparation-report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--freeze-reviewed", type=Path)
    args = parser.parse_args()
    if args.freeze_reviewed:
        print(json.dumps(freeze(args.freeze_reviewed), indent=2))
    else:
        prepare(args.tokenizer)
