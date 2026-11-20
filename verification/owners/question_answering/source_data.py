# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Pinned source-only QA preparation. No models or evaluation pool material."""
import argparse
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


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
        left, right = self.find(left), self.find(right)
        self.parent[max(left, right)] = min(left, right)


def eligible(row):
    if (row.get("lang") != "en" or row.get("synthetic") is not False
            or row.get("deleted") is not False or row.get("review_result") is not True
            or not isinstance(row.get("review_count"), int) or row["review_count"] <= 0
            or not isinstance(row.get("text"), str) or not row["text"].strip()):
        return False
    labels = row.get("labels")
    if isinstance(labels, dict):
        names, values = labels.get("name", []), labels.get("value", [])
        if len(names) != len(values):
            return False
        pairs = list(zip(names, values))
    elif isinstance(labels, list):
        pairs = [(x.get("name"), x.get("value")) for x in labels if isinstance(x, dict)]
    else:
        return False
    pii = [v for k, v in pairs if k == "pii"]
    return len(pii) == 1 and isinstance(pii[0], (int, float)) and math.isfinite(pii[0]) and pii[0] == 0


def group_messages(sources, candidates):
    """Whole trees stay disjoint. Only actual candidate input text links trees."""
    links, seen, memberships = Components(), {}, defaultdict(set)
    identifiers = set()
    for split, rows in sources.items():
        for row in rows:
            ident, tree = row["message_id"], row["message_tree_id"]
            if not isinstance(tree, str) or not tree or ident in identifiers:
                raise ValueError("invalid tree or duplicate message identity")
            identifiers.add(ident)
            links.find(tree)
    for row in candidates:
        tree = row["tree_id"]
        for field in ("prompt", "chosen", "rejected"):
            key = norm(row[field])
            if key in seen:
                links.union(tree, seen[key])
            seen[key] = tree
    for split, rows in sources.items():
        for row in rows:
            memberships[links.find(row["message_tree_id"])].add(split)
    cross = {key for key, values in memberships.items() if len(values) != 1}
    return links, cross


def preferences(sources, tokenize):
    groups, counts, candidates = defaultdict(list), Counter(), []
    for split, rows in sources.items():
        counts[split + "_messages"] = len(rows)
        children = defaultdict(list)
        byid = {row["message_id"]: row for row in rows}
        for row in rows:
            children[row["parent_id"]].append(row)
        for root in rows:
            if root["parent_id"] is not None or root["role"] != "prompter" or not eligible(root):
                continue
            replies = []
            for row in children[root["message_id"]]:
                rank = row["rank"]
                if (row["role"] == "assistant" and eligible(row) and type(rank) is int
                        and rank >= 0 and row["message_tree_id"] == root["message_tree_id"]):
                    replies.append(row)
            replies.sort(key=lambda row: (row["rank"], row["message_id"]))
            for i, chosen in enumerate(replies):
                for rejected in replies[i + 1:]:
                    if chosen["rank"] == rejected["rank"] or norm(chosen["text"]) == norm(rejected["text"]):
                        continue
                    counts[split + "_original_pairs"] += 1
                    tokens = [tokenize(root["text"], row["text"]) for row in (chosen, rejected)]
                    if max(map(len, tokens)) > 256 or tokens[0] == tokens[1]:
                        counts[split + "_invisible_pairs"] += 1
                        continue
                    assert root["message_id"] in byid
                    candidates.append({"prompt": root["text"], "chosen": chosen["text"],
                                          "rejected": rejected["text"],
                                          "tree_id": root["message_tree_id"], "root_id": root["message_id"],
                                          "chosen_id": chosen["message_id"], "rejected_id": rejected["message_id"],
                                          "chosen_rank": chosen["rank"], "rejected_rank": rejected["rank"],
                                          "source_split": split, "source_domain": "OpenAssistant/oasst1",
                                          "source_revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
                                          "source_license": "Apache-2.0", "response_authorship": "source synthetic=false",
                                          "prompt_authorship": "source synthetic=false", "annotation_authorship": "OASST human sibling rank",
                                          "label_provenance": "original human sibling rank",
                                          "label_kind": "preference_pair", "margin": 0.0})
    links, cross = group_messages(sources, candidates)
    for row in candidates:
        group = links.find(row["tree_id"])
        if group not in cross:
            groups[group].append({**row, "group_id": group})
    pools = defaultdict(list)
    for group, originals in sorted(groups.items()):
        # Pair identity only selects the original. No score or style-based retry.
        row = min(originals, key=lambda r: digest((r["root_id"] + r["chosen_id"] + r["rejected_id"]).encode()))
        split = "audit" if row["source_split"] == "validation" else (
            "validation" if int(digest(("dev42" + group).encode()), 16) % 5 == 0 else "train")
        direction = "longer" if len(row["chosen"]) > len(row["rejected"]) else (
            "shorter" if len(row["chosen"]) < len(row["rejected"]) else "tie")
        pools[split, direction].append(row)
    counts["cross_official_components_removed"] = len(cross)
    return pools, dict(counts)


def select(pools, limits):
    selected = {}
    for split in ("audit", "train", "validation"):
        count = limits[split]
        if count % 2 or min(len(pools[split, d]) for d in ("longer", "shorter")) < count // 2:
            raise ValueError("insufficient independent length-balanced groups: " + split)
        selected[split] = sorted(pools[split, "longer"][:count // 2] + pools[split, "shorter"][:count // 2],
                                 key=lambda row: row["group_id"])
    check_splits(selected)
    return selected


def check_splits(splits):
    all_groups, all_text, all_trees, all_renders = set(), set(), set(), set()
    for name, rows in splits.items():
        groups = {r["group_id"] for r in rows}
        text = {norm(r[k]) for r in rows for k in ("prompt", "chosen", "rejected")}
        trees = {r["tree_id"] for r in rows if "tree_id" in r}
        rendered = {norm(render(r["prompt"], r[side])) for r in rows for side in ("chosen", "rejected")}
        if len(groups) != len(rows) or groups & all_groups or text & all_text or trees & all_trees or rendered & all_renders:
            raise ValueError("connected component or normalized text leakage: " + name)
        if any(r["chosen_rank"] >= r["rejected_rank"] for r in rows):
            raise ValueError("invalid ascending source preference direction")
        all_groups.update(groups)
        all_text.update(text)
        all_trees.update(trees)
        all_renders.update(rendered)


def get_sources(out, cache=None):
    import time
    import urllib.request
    spec = json.loads((HERE / "sources.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in spec["files"].items():
        target = out / name
        cached = cache / name if cache else target
        if cached.is_file():
            if cached.stat().st_size > spec["limits"]["max_download_bytes"]:
                raise ValueError("cached source exceeds byte ceiling")
            raw = cached.read_bytes()
        else:
            start = time.monotonic()
            chunks, total = [], 0
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "QA-source-gate/1"}), timeout=45) as response:
                while chunk := response.read(1 << 20):
                    total += len(chunk)
                    if total > spec["limits"]["max_download_bytes"] or time.monotonic() - start > 180:
                        raise ValueError("source download ceiling")
                    chunks.append(chunk)
            raw = b"".join(chunks)
        if len(raw) > spec["limits"]["max_download_bytes"] or digest(raw) != sha:
            raise ValueError("source pin mismatch: " + name)
        target.write_bytes(raw)
    (out / "LICENSE").write_bytes((out / "oasst-license.txt").read_bytes())
    (out / "NOTICE").write_text((HERE / "NOTICE").read_text())
    return spec


def prepare(out, cache=None):
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    spec = get_sources(out / "sources", cache)
    tokenizer = Tokenizer.from_file(str(out / "sources/tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    sources = {}
    columns = ["message_id", "parent_id", "message_tree_id", "text", "role", "lang", "rank",
               "review_count", "review_result", "deleted", "synthetic", "labels"]
    for split in ("train", "validation"):
        table = pq.read_table(out / "sources" / ("oasst-" + split + ".parquet"), columns=columns)
        if table.num_rows > spec["limits"]["max_messages"]:
            raise ValueError("source row ceiling")
        sources[split] = table.to_pylist()
    pools, counts = preferences(sources, lambda p, a: tokenizer.encode(render(p, a)).ids)
    available = {s: {d: len(pools[s, d]) for d in ("longer", "shorter", "tie")} for s in ("audit", "train", "validation")}
    write_json(out / "preselection-report.json", {"counts": counts, "available_groups": available,
               "requested": {s: spec["limits"][s] for s in available}, "status": "SOURCE_ONLY"})
    try:
        selected = select(pools, spec["limits"])
        complete = True
        (out / "selection-failure.json").unlink(missing_ok=True)
    except ValueError as exc:
        # Preserve a labelled prototype for data gates only. Never lower a real
        # training requirement or use the audit to select a different recipe.
        selected = {}
        for split in ("audit", "train", "validation"):
            half = min(spec["limits"][split] // 2, len(pools[split, "longer"]), len(pools[split, "shorter"]))
            selected[split] = sorted(pools[split, "longer"][:half] + pools[split, "shorter"][:half], key=lambda r: r["group_id"])
        check_splits(selected)
        complete = False
        write_json(out / "selection-failure.json", {"status": "INCOMPLETE", "failure": str(exc),
                   "training_allowed": False, "prototype_only": True})
    files = {}
    for split in ("audit", "train", "validation"):
        raw = "".join(json.dumps(r, ensure_ascii=True, separators=(",", ":")) + "\n" for r in selected[split]).encode()
        (out / (split + ".jsonl")).write_bytes(raw)
        files[split] = {"rows": len(selected[split]), "groups": len(selected[split]), "sha256": digest(raw),
                        "unique_negative_fraction": len({norm(r["rejected"]) for r in selected[split]}) / len(selected[split]) if selected[split] else None}
    report = {"status": "SOURCE_ONLY" if complete else "INCOMPLETE", "training_allowed": complete,
              "prototype_only": not complete, "scope": spec["scope"], "source_pins": spec,
              "counts": counts, "files": files,
              "available_groups": {s: {d: len(pools[s, d]) for d in ("longer", "shorter", "tie")} for s in selected},
               "grouping": "Whole source trees grouped by normalized actual compact root question/chosen/rejected inputs. Unused descendants never link unrelated roots. Cross-official components removed.",
              "final_audit_frozen_before_training": True, "original_text_and_rank_unchanged": True,
               "privacy": "Explicit source PII=0 required. This is not an independent privacy review.",
               "modification": "English root sibling filtering, component selection and fixed length-balanced sampling"}
    overlaps = {}
    for left, right in (("train", "validation"), ("train", "audit"), ("validation", "audit")):
        def texts(name):
            return {norm(r[k]) for r in selected[name] for k in ("prompt", "chosen", "rejected")}
        def rendered(name):
            return {norm(render(r["prompt"], r[k])) for r in selected[name] for k in ("chosen", "rejected")}
        overlaps[left + "/" + right] = {"normalized_fields": len(texts(left) & texts(right)),
                                        "normalized_renders": len(rendered(left) & rendered(right)),
                                        "whole_trees": len({r["tree_id"] for r in selected[left]} & {r["tree_id"] for r in selected[right]})}
    report["cross_split_overlap"] = overlaps
    token_binding = {}
    for split, rows in selected.items():
        token_binding[split] = [{"group_id": row["group_id"], "root_id": row["root_id"],
                                "chosen_id": row["chosen_id"], "rejected_id": row["rejected_id"],
                                **{side: {"render_sha256": digest(render(row["prompt"], row[side]).encode()),
                                           "token_sha256": digest(json.dumps(tokenizer.encode(render(row["prompt"], row[side])).ids).encode()),
                                           "tokens": len(tokenizer.encode(render(row["prompt"], row[side])).ids)}
                                   for side in ("chosen", "rejected")}} for row in rows]
    write_json(out / "token-visible-binding.json", {"SPDX-License-Identifier": "Apache-2.0", "splits": token_binding,
               "max_tokens": 256, "cropped": False, "distinct_pairs": sum(len(rows) for rows in selected.values())})
    write_json(out / "data-report.json", report)
    (out / "LICENSE").write_bytes((out / "sources/LICENSE").read_bytes())
    (out / "NOTICE").write_text((HERE / "NOTICE").read_text())
    return report


def render(prompt, answer):
    return f"### Task:\n{prompt}\n\n### Answer:\n{answer}"


def read_splits(out):
    report = json.loads((out / "data-report.json").read_text())
    splits = {}
    for split, pin in report["files"].items():
        raw = (out / (split + ".jsonl")).read_bytes()
        if len(raw) > 8 << 20 or digest(raw) != pin["sha256"]:
            raise ValueError("split pin mismatch")
        splits[split] = [json.loads(line) for line in raw.decode().split("\n") if line]
        if len(splits[split]) != pin["rows"]:
            raise ValueError("split count mismatch")
    check_splits(splits)
    return report, splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-cache", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, args.source_cache), indent=2))


if __name__ == "__main__":
    main()
