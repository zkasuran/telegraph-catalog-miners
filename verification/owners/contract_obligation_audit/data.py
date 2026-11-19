# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Complete licensed-source scan and frozen grouped clause-entailment pairs."""
import json
import re
import time
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict
from urllib.parse import urlparse

from contracts import LABELS, file_sha, sha, strict_json

RIGHTS_FLAGS = re.compile(r"all rights reserved|not (?:be )?reproduced|reproduction (?:is )?prohibited|permission (?:of|from).*?(?:reproduc|publish)", re.I)


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def download(url, path, expected, ceiling=70 << 20):
    if path.exists() and file_sha(path) == expected:
        return
    start = time.monotonic()
    request = urllib.request.Request(url, headers={"User-Agent": "contract-obligation-source-pipeline/1.0"})
    partial = path.with_suffix(".partial")
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=45) as stream, partial.open("wb") as out:
            while chunk := stream.read(1 << 20):
                total += len(chunk)
                if total > ceiling or time.monotonic() - start > 300:
                    raise ValueError("source download size/time ceiling")
                out.write(chunk)
        if file_sha(partial) != expected:
            raise ValueError("download source hash mismatch")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def source_group(doc):
    parsed = urlparse(doc["url"])
    cik = re.search(r"/edgar/data/(\d+)/", parsed.path, re.I)
    return "sec-cik:" + cik[1] if cik else "source-host:" + (parsed.hostname or "missing").lower().removeprefix("www.")


def pair(row, style="json"):
    def answer(relation):
        if style == "json":
            return json.dumps({"verdict": "Unclear", "obligations": [{"claim": row["hypothesis"], "status": relation}], "clauses": [row["evidence"]],
                               "summary": "The clause " + relation + " this claim: " + row["hypothesis"] + " Performance is not established."},
                              ensure_ascii=False, separators=(",", ":"))
        if style == "prose":
            return "Verdict: Unclear. The clause " + relation + " this claim: " + row["hypothesis"] + " Clause: " + row["evidence"] + " Performance is not established."
        raise ValueError("unclaimed response format")
    true = "supports" if row["label"] == "Entailment" else "refutes"
    false = "refutes" if true == "supports" else "supports"
    return {**row, "prompt": row["evidence"], "chosen": answer(true), "rejected": answer(false),
            "label_kind": "preference_pair", "margin": 0.0,
            "authorship": "Published document-level NLI/evidence labels, AI-assisted response templates; not fresh human preference reviews"}


def collect_source(archive_path, config):
    records, annotations, counts = [], {}, Counter()
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if len(infos) > 5000 or sum(i.file_size for i in infos) > 300 << 20:
            raise ValueError("source archive ceiling")
        # Read all source members, not just a population prefix.
        for info in infos:
            if not info.is_dir():
                if info.file_size > 32 << 20:
                    raise ValueError("source member ceiling")
                archive.read(info)
                counts["members_read"] += 1
        labels = None
        for original_split, split in (("train", "train"), ("dev", "validation"), ("test", "audit")):
            obj = strict_json(archive.read("contract-nli/" + original_split + ".json"))
            if labels is not None and labels != obj["labels"]:
                raise ValueError("source hypotheses changed across splits")
            labels = obj["labels"]
            for doc in obj["documents"]:
                text, spans = doc["text"], doc["spans"]
                for start, end in spans:
                    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                        raise ValueError("source span offset invalid")
                if len(doc["annotation_sets"]) != 1:
                    raise ValueError("source annotation count changed")
                anno = doc["annotation_sets"][0]["annotations"]
                if set(anno) != set(labels):
                    raise ValueError("incomplete source annotation inventory")
                key = str(doc["id"])
                if key in annotations:
                    raise ValueError("duplicate document id")
                annotations[key] = doc
                flags = bool(RIGHTS_FLAGS.search(text))
                record = {"document_id": key, "document_sha256": sha(text.encode()),
                          "normalized_document_sha256": sha(normalized(text).encode()), "source_url": doc["url"],
                          "source_group": source_group(doc), "split": split, "rights_deferred": flags}
                records.append(record)
                counts["documents"] += 1
                counts["sentence_spans"] += len(spans)
                counts["rights_deferred_documents"] += flags
                for annotation in anno.values():
                    choice, indices = annotation["choice"], annotation["spans"]
                    if choice not in (*LABELS, "NotMentioned"):
                        raise ValueError("source choice invalid")
                    if len(indices) != len(set(indices)) or any(type(i) is not int or not 0 <= i < len(spans) for i in indices):
                        raise ValueError("source evidence indices invalid")
                    if (choice == "NotMentioned" and indices) or (choice in LABELS and not indices):
                        raise ValueError("source label/evidence inconsistency")
                    counts["annotations"] += 1
    if counts["documents"] != 607 or counts["annotations"] != 10319:
        raise ValueError("complete published population count mismatch")
    # Join exact-normalized originals and the same source/entity host before any selection.
    parent = {r["document_id"]: r["document_id"] for r in records}
    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    previous = {}
    for row in records:
        for key in (row["normalized_document_sha256"], row["source_group"]):
            if key in previous:
                parent[root(row["document_id"])] = root(previous[key])
            else:
                previous[key] = row["document_id"]
    components = defaultdict(list)
    for row in records:
        components[root(row["document_id"])].append(row)
    groups = {}
    for members in components.values():
        identity = sha("\n".join(sorted(r["document_sha256"] for r in members)).encode())
        splits = {r["split"] for r in members}
        for row in members:
            groups[row["document_id"]] = (identity, len(splits) > 1)
    eligible = defaultdict(list)
    overlap = defaultdict(set)
    for record in records:
        group, crossing = groups[record["document_id"]]
        if crossing or record["rights_deferred"]:
            counts["documents_removed_group_or_rights"] += 1
            continue
        doc = annotations[record["document_id"]]
        for key, annotation in doc["annotation_sets"][0]["annotations"].items():
            if annotation["choice"] not in LABELS:
                continue  # No excerpt-based absence claims.
            indices = sorted(annotation["spans"])
            evidence = "\n".join(doc["text"][doc["spans"][i][0]:doc["spans"][i][1]] for i in indices)
            if not 32 <= len(evidence.encode()) <= config["max_evidence_bytes"]:
                continue
            # Incomplete list/conjunction fragments do not become local entailment claims.
            if re.search(r"(?:\bor|\band|:|;)\s*$", evidence, re.I):
                continue
            identity = sha(normalized(evidence).encode())
            overlap[identity].add(record["split"])
            eligible[record["split"]].append({**record, "source_group": group, "evidence": evidence,
                "evidence_sha256": sha(evidence.encode()), "normalized_evidence_sha256": identity,
                "source_evidence_indices": indices, "hypothesis_id": key, "hypothesis": labels[key]["hypothesis"],
                "label": annotation["choice"], "source_label_scope": "document-scoped; every annotated evidence span retained"})
    conflicts = {key for key, splits in overlap.items() if len(splits) > 1}
    eligible = {split: [r for r in rows if r["normalized_evidence_sha256"] not in conflicts]
                for split, rows in eligible.items()}
    for record in records:
        record["connected_group"], record["cross_official_split"] = groups[record["document_id"]]
    return eligible, {"counts": dict(counts), "documents": records, "normalized_overlap_removed": len(conflicts),
                      "source_grant": "ContractNLI publisher CC BY 4.0, original grants retained; flagged originals deferred",
                      "audit_used_for_selection": False, "normalized_overlap": 0}


def scan(archive_path, config):
    raise ValueError("unbalanced source generations are retired; use balanced.construct")


def check_splits(splits, config):
    seen_groups, seen_evidence = set(), set()
    for split, rows in splits.items():
        if len(rows) != config["limits"][split] or Counter(r["label"] for r in rows) != {label: len(rows)//2 for label in LABELS}:
            raise ValueError("frozen split size/label balance mismatch")
        groups = {r["source_group"] for r in rows}
        evidence = {r["normalized_evidence_sha256"] for r in rows}
        if seen_groups & groups or seen_evidence & evidence or len(evidence) != len(rows):
            raise ValueError("cross-split source group or normalized evidence overlap")
        seen_groups |= groups
        seen_evidence |= evidence
        if split == "audit" and len(groups) < 32:
            raise ValueError("audit requires at least 32 independent source groups")
        for row in rows:
            if (row["prompt"] != row["evidence"] or sha(row["evidence"].encode()) != row["evidence_sha256"]
                    or len(row["evidence"].encode()) > config["max_evidence_bytes"] or row["chosen"] == row["rejected"]):
                raise ValueError("source/pair identity or input bounds mismatch")


def main():
    from balanced import main as balanced_main
    return balanced_main()


if __name__ == "__main__":
    main()
