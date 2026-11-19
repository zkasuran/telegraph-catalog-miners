# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Enumerate the full pinned article inventory then collect a bounded pilot.

Only public text, metadata and small generator configs are fetched locally.
No generator weight URL is fetched by this program.
"""
import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from common import COUNTS, HERE, dump, fetch, identity, sha

REVISION = "c3cd502f94edfb5aa0c5870ed9eff053773c6125"
TREE = "4ee79e98a7dbdcd3dd040be4b836e0843520c1fd"
RAW = f"https://raw.githubusercontent.com/elifesciences/elife-article-xml/{REVISION}/articles/"
XLINK = "{http://www.w3.org/1999/xlink}href"
LICENSES = {"http://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
            "http://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0"}
TERMS = {
    "cc_by_3": "https://creativecommons.org/licenses/by/3.0/legalcode.txt",
    "cc_by_4": "https://creativecommons.org/licenses/by/4.0/legalcode.txt",
    "apache": "https://www.apache.org/licenses/LICENSE-2.0.txt",
    "gpt2": "https://raw.githubusercontent.com/openai/gpt-2/9b63575ef42771a015060c964af2c3da4cf7c8ab/LICENSE",
    "kit": "https://raw.githubusercontent.com/telegraphprotocol/telegraph-wasm-baseline/1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd/LICENSE",
}
GENERATORS = {
    "gpt2": {"id": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
             "weight_sha256": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
             "weight_bytes": 548105171, "licence": "OpenAI-Modified-MIT"},
    "distilgpt2": {"id": "distilbert/distilgpt2", "revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
                   "weight_sha256": "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b",
                   "weight_bytes": 352824413, "licence": "Apache-2.0"},
}


def parse_article(raw, entry):
    git_sha = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if git_sha != entry["sha"] or len(raw) != entry["size"]:
        raise ValueError("article differs from pinned git tree")
    if b"<!ENTITY" in raw or len(raw) > 2 << 20:
        raise ValueError("XML entity or size ceiling")
    root = ET.fromstring(raw)
    if root.get("article-type") != "research-article":
        raise ValueError("not a research article")
    meta = root.find("./front/article-meta")
    licence = meta.find("./permissions/license")
    if licence is None:
        raise ValueError("no individual source grant")
    uri = licence.get(XLINK, "").replace("https://", "http://")
    if uri not in LICENSES:
        raise ValueError("source grant not cleared")
    clause = " ".join("".join(licence.itertext()).split())
    if "unrestricted" not in clause or "credited" not in clause:
        raise ValueError("grant clause missing")
    dates = [p for p in meta.findall("pub-date") if p.get("pub-type") == "epub" or p.get("publication-format") == "electronic"]
    if not dates:
        raise ValueError("no electronic publication year")
    year = int(dates[0].findtext("year", "0"))
    split = "train" if 2012 <= year <= 2014 else "validation" if year == 2015 else "test" if 2016 <= year <= 2018 else None
    if split is None:
        raise ValueError("outside frozen pre-2019 time blocks")
    authors = [" ".join(" ".join(c.find("name").itertext()).split())
               for c in meta.findall("./contrib-group/contrib")
               if c.get("contrib-type") == "author" and c.find("name") is not None]
    if not authors:
        raise ValueError("no source author attribution")
    doi = next((a.text for a in meta.findall("article-id") if a.get("pub-id-type") == "doi"), None)
    if not doi:
        raise ValueError("no source DOI")
    # Exclude references, figures, tables and attributed quoted blocks. Extract a
    # contiguous word span from the first plain abstract paragraph only.
    paragraphs = [p for p in meta.findall("./abstract/p")
                  if not any(p.find(".//" + tag) is not None for tag in ("xref", "ext-link", "disp-quote", "fig", "table-wrap"))]
    if not paragraphs:
        raise ValueError("no plain abstract paragraph")
    paragraph = " ".join("".join(paragraphs[0].itertext()).split())
    words = list(re.finditer(r"\S+", paragraph))
    if len(words) < 37:
        raise ValueError("short original paragraph")
    prefix = paragraph[:words[11].end()]
    human = paragraph[words[12].start():words[35].end()]
    if not 32 <= len(human.encode()) <= 256:
        raise ValueError("excerpt exceeds literal request limit")
    return {"source_id": doi, "group_id": identity(doi), "split": split, "year": year,
            "path": entry["path"], "git_blob": git_sha, "sha256": sha(raw), "bytes": len(raw),
            "url": RAW + entry["path"], "licence": LICENSES[uri], "licence_url": uri,
            "grant_clause": clause, "authors": authors, "author_ids": [identity(a) for a in authors],
            "title": " ".join("".join(meta.find("./title-group/article-title").itertext()).split()),
            "paragraph": paragraph, "prefix": prefix, "human_text": human,
            "human_span": [words[12].start(), words[35].end()],
            "origin_basis": "Pre-2019 journal text attributed to named authors. Human-control origin is an assumption, not certified production history.",
            "changes": "XML markup removed and whitespace collapsed; contiguous 24-word abstract excerpt and 12-word generation prefix selected."}


def collect(out):
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "xml"
    cache.mkdir(exist_ok=True)
    started = time.monotonic()
    raw = fetch(f"https://api.github.com/repos/elifesciences/elife-article-xml/git/trees/{TREE}", out / "inventory.json")
    tree = json.loads(raw)
    if tree.get("sha") != TREE or tree.get("truncated") is not False or any(r["type"] == "tree" for r in tree["tree"]):
        raise ValueError("incomplete inventory")
    entries = [r for r in tree["tree"] if r["type"] == "blob" and re.fullmatch(r"elife-\d+-v\d+\.xml", r["path"])]
    all_groups = {r["path"].split("-v")[0] for r in entries}
    # Population counts come from the full tree, not this eligible candidate set.
    candidates = [r for r in entries if r["path"].endswith("-v1.xml") and int(r["path"].split("-")[1]) <= 30000
                  and r["path"] not in ("elife-00666-v1.xml", "elife-45374-v1.xml")]
    candidates.sort(key=lambda r: sha(("ai-origin-pilot-v1:" + r["path"]).encode()))
    selected = {s: [] for s in COUNTS}
    excluded, seen_authors, seen_text, seen_doi, scanned, failures = Counter(), set(), set(), set(), 0, []
    def get(entry):
        try:
            path = cache / entry["path"]
            raw = path.read_bytes() if path.is_file() and path.stat().st_size <= 2 << 20 else fetch(RAW + entry["path"], path, ceiling=2 << 20)
            return parse_article(raw, entry), None
        except (ValueError, OSError, ET.ParseError) as e:
            return None, str(e)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, min(len(candidates), 2048), 32):
            if time.monotonic() - started > 1100:
                raise ValueError("source preparation time ceiling")
            batch = candidates[start:start + 32]
            for entry, (row, error) in zip(batch, pool.map(get, batch)):
                scanned += 1
                if error:
                    excluded[error] += 1
                    failures.append({"path": entry["path"], "reason": error})
                    continue
                split = row["split"]
                if len(selected[split]) >= COUNTS[split]:
                    excluded["split capacity"] += 1
                    continue
                ids = {identity(row["prefix"]), identity(row["human_text"]), identity(row["paragraph"])}
                if seen_authors.intersection(row["author_ids"]) or seen_text.intersection(ids) or row["source_id"] in seen_doi:
                    excluded["source author/DOI/normalized text collision"] += 1
                    continue
                selected[split].append(row)
                seen_authors.update(row["author_ids"])
                seen_text.update(ids)
                seen_doi.add(row["source_id"])
            progress = {"phase": "source", "scanned": scanned, "selected": {s: len(rs) for s, rs in selected.items()}}
            dump(out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
            if all(len(selected[s]) == n for s, n in COUNTS.items()):
                break
    report = {"schema_version": 1, "revision": REVISION, "article_tree": TREE,
              "inventory_sha256": sha(raw), "inventory_bytes": len(raw), "inventory_truncated": False,
              "inventory_xml_files": len(entries), "inventory_article_groups": len(all_groups),
              "candidate_pool": len(candidates), "scanned": scanned, "excluded": dict(excluded),
              "selection": "Fixed hash-order bounded sample from complete v1 inventory under article ID 30000. Not a population quality benchmark.",
              "counts": {s: len(rs) for s, rs in selected.items()}, "articles": selected,
              "normalized_source_overlap": 0, "cross_split_named_author_overlap": 0}
    dump(out / "source-report.json", report)
    dump(out / "source-exclusions.json", failures)
    if report["counts"] != COUNTS:
        raise ValueError("insufficient source groups; report retained")
    return report


def pins(out, report, licence_file):
    terms = {}
    for name, url in TERMS.items():
        raw = fetch(url, out / (name + "-terms.txt"), ceiling=64 << 10)
        terms[name] = {"url": url, "sha256": sha(raw), "text": raw.decode()}
    # Take the reviewed SAND text verbatim, never re-draft it.
    upstream_terms = json.loads(licence_file.read_text())
    terms["sand"] = {"text": upstream_terms["sand"], "scope": "Only our competitive additions. Source grants survive."}
    generators = {}
    for name, spec in GENERATORS.items():
        files = {"model.safetensors": spec["weight_sha256"]}
        for filename in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "README.md"):
            url = f"https://huggingface.co/{spec['id']}/resolve/{spec['revision']}/{filename}"
            raw = fetch(url, out / (name + "-" + filename), ceiling=2 << 20)
            files[filename] = sha(raw)
        generators[name] = {**spec, "files": files}
    return {"schema_version": 1, "SPDX-License-Identifier": "LicenseRef-zkasuran-SAND-1.0",
            "source_report": report, "generators": generators, "terms": terms}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--licences", type=Path, default=HERE / "licensed-components.json")
    args = p.parse_args()
    report = collect(args.out)
    dump(args.release, pins(args.out, report, args.licences))
    print(json.dumps({"source_groups": sum(COUNTS.values()), "release_sha256": sha(args.release.read_bytes()),
                      "generator_weights_downloaded": False}))


if __name__ == "__main__":
    main()
