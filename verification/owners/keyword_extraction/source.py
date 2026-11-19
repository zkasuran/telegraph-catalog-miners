# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded public eLife source processing. No model or protected-pool access."""
import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

REVISION = "c3cd502f94edfb5aa0c5870ed9eff053773c6125"
BASE = f"https://raw.githubusercontent.com/elifesciences/elife-article-xml/{REVISION}/articles/"
MAX_FILE = 4 << 20
MAX_TOTAL = 512 << 20
MAX_RECORDS = 4096
XLINK = "{http://www.w3.org/1999/xlink}href"
GRANTS = {
    "https://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
    "https://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
    "https://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def contains(passage, phrase):
    return re.search(r"(?<!\w)" + re.escape(norm(phrase)) + r"(?!\w)", norm(passage)) is not None


def text(node):
    return "" if node is None else "".join(node.itertext()).strip()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def verify_tree(tree):
    if tree.get("truncated") is not False:
        raise ValueError("inventory truncated or completeness unknown")
    parts = []
    entries = sorted(tree["tree"], key=lambda e: (e["path"] + ("/" if e["type"] == "tree" else "")).encode())
    for entry in entries:
        if "/" in entry["path"] or "\0" in entry["path"]:
            raise ValueError("expected a direct article subtree, not a recursive path list")
        parts.append(entry["mode"].lstrip("0").encode() + b" " + entry["path"].encode() + b"\0" + bytes.fromhex(entry["sha"]))
    payload = b"".join(parts)
    calculated = hashlib.sha1(b"tree " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
    if calculated != tree["sha"]:
        raise ValueError("full source inventory Git tree hash mismatch")
    return calculated


def fetch(url, ceiling=MAX_FILE):
    started = time.monotonic()
    request = urllib.request.Request(url, headers={"User-Agent": "Telegraph-keyword-source/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(ceiling + 1)
        if len(raw) > ceiling or time.monotonic() - started > 90:
            raise ValueError("source exceeds byte/time ceiling")
        return raw


def parse(raw, entry):
    if len(raw) > MAX_FILE or b"<!ENTITY" in raw.upper():
        raise ValueError("oversize XML or entity declaration")
    if blob(raw) != entry["sha"]:
        raise ValueError("pinned git blob mismatch")
    doc = ET.fromstring(raw)
    meta = doc.find("./front/article-meta")
    if meta is None or doc.get("article-type") != "research-article":
        raise ValueError("not an original research article")
    licenses = meta.findall("./permissions/license")
    if len(licenses) != 1:
        raise ValueError("missing or ambiguous article grant")
    uri = licenses[0].get(XLINK, "").replace("http://", "https://").rstrip("/") + "/"
    if uri not in GRANTS or not text(licenses[0]):
        raise ValueError("unreviewed article grant")
    abstracts = [x for x in meta.findall("abstract") if not x.get("abstract-type")]
    if len(abstracts) != 1:
        raise ValueError("missing or ambiguous primary abstract")
    abstract = abstracts[0]
    if abstract.find(".//attrib") is not None or abstract.find(".//graphic") is not None:
        raise ValueError("abstract has separately credited or graphical content")
    # JATS object-id and a standalone DOI footer are not prose. Keep every p
    # and all inline scientific text, including formulas and Unicode.
    paragraphs = []
    for p in abstract.iter("p"):
        if p.find(".//p") is not None:
            continue
        value = text(p)
        if re.fullmatch(r"DOI:\s*https?://(?:dx\.)?doi.org/\S+", value):
            continue
        if value:
            paragraphs.append(value)
    keywords = [text(k) for g in meta.findall("kwd-group")
                if g.get("kwd-group-type") == "author-keywords" for k in g.findall("kwd")]
    doi = text(meta.find("article-id[@pub-id-type='doi']"))
    title = text(meta.find("./title-group/article-title"))
    if not doi.startswith("10.7554/") or not title or not paragraphs or not keywords:
        raise ValueError("missing original DOI/title/abstract/author-keywords")
    authors = [text(c.find("name/given-names")) + " " + text(c.find("name/surname"))
               for c in meta.findall("./contrib-group/contrib") if c.get("contrib-type") == "author"]
    authors = [a.strip() for a in authors if a.strip()]
    if not authors:
        raise ValueError("missing creator attribution")
    topics = sorted({text(s) for g in meta.findall("./article-categories/subj-group")
                     if g.get("subj-group-type") in {"heading", "sub-display-channel"}
                     for s in g.findall("subject") if text(s)})
    year = text(meta.find("pub-date[@date-type='pub']/year")) or text(meta.find("pub-date/year"))
    passage = title + "\n\n" + "\n\n".join(paragraphs)
    return {"doi": doi, "version": 1, "source_path": entry["path"], "url": BASE + entry["path"],
            "git_blob": entry["sha"], "sha256": sha(raw), "title": title,
            "abstract": "\n\n".join(paragraphs), "text": passage,
            "author_keywords": keywords, "authors": authors, "topics": topics, "year": year,
            "licence": GRANTS[uri], "licence_uri": uri, "grant": text(licenses[0]),
            "copyright": text(meta.find("permissions/copyright-statement")),
            "source_elements": ["front/article-meta/title-group/article-title", "front/article-meta/abstract/p",
                                "front/article-meta/kwd-group[@kwd-group-type='author-keywords']/kwd"],
            "modifications": "Inline XML flattened; paragraph boundaries retained; JATS IDs and standalone DOI footer omitted. No prose cropping.",
            "literal_keywords": [k for k in keywords if contains(passage, k)],
            "annotation_kind": "original author keywords, not exhaustive preference gold"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--reuse", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", type=int, default=MAX_RECORDS)
    args = parser.parse_args()
    if not 1 <= args.count <= MAX_RECORDS:
        raise ValueError("record ceiling exceeded")
    args.out.mkdir(parents=True, exist_ok=True)
    raw_inventory = args.inventory.read_bytes()
    tree = json.loads(raw_inventory)
    verify_tree(tree)
    commit_url = f"https://api.github.com/repos/elifesciences/elife-article-xml/git/commits/{REVISION}"
    commit = json.loads(fetch(commit_url, 1 << 20))
    parent_url = "https://api.github.com/repos/elifesciences/elife-article-xml/git/trees/" + commit["tree"]["sha"]
    parent = json.loads(fetch(parent_url, 1 << 20))
    article = next(e for e in parent["tree"] if e["path"] == "articles")
    if commit["sha"] != REVISION or article["sha"] != tree["sha"]:
        raise ValueError("inventory does not belong to pinned eLife commit")
    write(args.out / "inventory-binding.json", {"commit_url": commit_url, "parent_url": parent_url,
          "commit": commit, "parent_tree": parent, "article_tree": tree["sha"], "inventory_sha256": sha(raw_inventory),
          "full_git_tree_verified": True})
    entries = [e for e in tree["tree"] if e["type"] == "blob" and re.fullmatch(r"elife-\d+-v1.xml", e["path"])]
    if len({e["path"] for e in entries}) != len(entries):
        raise ValueError("duplicate inventory path")
    selected = sorted(entries, key=lambda e: sha(e["path"].encode()))[:args.count]
    if sum(e.get("size", MAX_FILE + 1) for e in selected) > MAX_TOTAL or any(e.get("size", MAX_FILE + 1) > MAX_FILE for e in selected):
        raise ValueError("selected inventory exceeds source byte ceiling")
    (args.out / "inventory.json").write_bytes(raw_inventory)
    write(args.out / "selection.json", {"revision": REVISION, "inventory_sha256": sha(raw_inventory),
          "inventory_entries": len(tree["tree"]), "all_v1_xml": len(entries), "selected": selected,
          "selection": "Filename SHA-256 ordering over the complete nontruncated v1 inventory; bounded sample, not the population."})
    def obtain(entry):
        target = args.out / entry["path"]
        cached = args.reuse / ("sample-" + entry["path"])
        try:
            raw = target.read_bytes() if target.is_file() else cached.read_bytes() if cached.is_file() else fetch(BASE + entry["path"])
            if blob(raw) != entry["sha"]:
                raise ValueError("pinned git blob mismatch")
            if len(raw) != entry["size"]:
                raise ValueError("inventory byte-size mismatch")
            if not target.exists():
                target.write_bytes(raw)
            return entry, raw, None
        except (OSError, ValueError) as exc:
            return entry, b"", str(exc)
    records, errors, counts, total = [], [], Counter(), 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for entry, raw, failure in executor.map(obtain, selected):
            total += len(raw)
            if total > MAX_TOTAL:
                raise ValueError("total source ceiling exceeded")
            if failure:
                errors.append({"path": entry["path"], "error": failure})
                counts["download_failed"] += 1
            else:
                counts["blob_verified"] += 1
                try:
                    records.append(parse(raw, entry))
                    counts["annotated_research_articles"] += 1
                except (ValueError, ET.ParseError) as exc:
                    errors.append({"path": entry["path"], "error": str(exc)})
                    counts[str(exc)] += 1
            write(args.out / "progress.json", {"counts": dict(counts), "bytes": total, "errors": errors})
    (args.out / "articles.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    write(args.out / "source-report.json", {"revision": REVISION, "counts": dict(counts), "bytes": total,
          "articles_sha256": sha((args.out / "articles.jsonl").read_bytes()), "errors": errors,
          "source_grants": dict(Counter(r["licence"] for r in records)), "scope": "Bounded sample of intact eLife research abstracts with whole original author-keyword annotations. Not preferences."})
    print(json.dumps({"counts": dict(counts), "bytes": total}))


if __name__ == "__main__":
    main()
