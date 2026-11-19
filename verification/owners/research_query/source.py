# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Bounded public originals only. No credentials, pool material or models."""
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
from datetime import datetime, timezone
from pathlib import Path

REVISION = "c3cd502f94edfb5aa0c5870ed9eff053773c6125"
TREE = "4ee79e98a7dbdcd3dd040be4b836e0843520c1fd"
KIT = "1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd"
TOKENIZER_SHA = "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456"
RAW = "https://raw.githubusercontent.com/elifesciences/elife-article-xml/" + REVISION + "/articles/"
MAX_XML = 4 << 20
MAX_PROMPT = 256
OPPOSITES = {"increased": "decreased", "decreased": "increased",
             "enhanced": "reduced", "reduced": "enhanced",
             "activated": "inhibited", "inhibited": "activated"}
EFFECT = re.compile(r"(?<![\w-])(" + "|".join(OPPOSITES) + r")(?![\w-])")
DISALLOWED = re.compile(r"\b(not|no|neither|nor|may|might|could|would|however|although|if|but|than|without|suggest|suggests)\b", re.I)
RIGHTS = {"http://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
          "https://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
          "http://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
          "https://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
          "http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
          "https://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


def read_url(url, ceiling):
    # Callers supply only fixed public hosts. Never use environment credentials.
    if not url.startswith(("https://raw.githubusercontent.com/", "https://api.github.com/",
                           "https://devnode.telegraphprotocol.com/", "https://creativecommons.org/")):
        raise ValueError("source host not allowed")
    start = time.monotonic()
    req = urllib.request.Request(url, headers={"User-Agent": "research-query-source-proof/1"})
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read(ceiling + 1)
        if response.status != 200 or len(raw) > ceiling or time.monotonic() - start > 60:
            raise ValueError("source status, size or elapsed ceiling")
        receipt = {"url": url, "final_url": response.geturl(), "sha256": sha(raw),
                   "bytes": len(raw), "fetched_at": datetime.now(timezone.utc).isoformat()}
    return raw, receipt


def text(element):
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def sentence_candidates(raw, path, blob_sha):
    if len(raw) > MAX_XML or b"<!ENTITY" in raw.upper():
        raise ValueError("XML ceiling or entity declaration")
    if hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != blob_sha:
        raise ValueError("primary Git blob identity mismatch")
    root = ET.fromstring(raw)
    if root.get("article-type") != "research-article":
        return [], "not_research_article"
    meta = root.find("./front/article-meta")
    if meta is None:
        return [], "no_article_metadata"
    licences = meta.findall("./permissions/license")
    if len(licences) != 1:
        return [], "ambiguous_licence"
    uri = licences[0].get("{http://www.w3.org/1999/xlink}href", "")
    if uri not in RIGHTS:
        return [], "uncleared_licence"
    doi = [text(e) for e in meta.findall("article-id") if e.get("pub-id-type") == "doi" and e.get("specific-use") != "version"]
    if len(doi) != 1 or not re.fullmatch(r"10\.7554/eLife\.\d+", doi[0], re.I):
        return [], "invalid_doi"
    authors = [text(e.find("name")) for e in meta.findall("./contrib-group/contrib") if e.get("contrib-type") == "author"]
    if not authors or not all(authors):
        return [], "missing_attribution"
    source = {"source_domain": "eLife", "source_revision": REVISION, "source_path": path,
              "source_url": RAW + path, "source_doi": norm(doi[0]), "source_version": "v1 XML",
              "source_sha256": sha(raw), "git_blob_sha1": blob_sha, "source_license": RIGHTS[uri],
              "license_uri": uri, "license_text": text(licences[0]), "authors": authors,
              "article_title": text(meta.find("./title-group/article-title")),
              "copyright_statement": text(meta.find("./permissions/copyright-statement")),
              "source_annotation": "none; original article text only"}
    candidates = []
    # Abstract only. No figures, captions, tables, references or external evidence.
    for ai, abstract in enumerate(meta.findall("abstract")):
        if abstract.get("abstract-type") or abstract.get("specific-use"):
            continue
        for pi, paragraph in enumerate(abstract.iter("p")):
            if any(e.tag in ("xref", "ext-link", "inline-formula", "disp-formula", "graphic") for e in paragraph.iter()):
                continue
            for si, sentence in enumerate(re.split(r"(?<=[.!?])\s+(?=[A-Z])", text(paragraph))):
                matches = list(EFFECT.finditer(sentence))
                if (len(matches) != 1 or DISALLOWED.search(sentence) or sentence[-1:] != "."
                        or not sentence[:1].isupper() or len(sentence) < 30):
                    continue
                match = matches[0]
                # Compound names and adjectival states are not direction claims.
                if match.group() in ("activated", "inhibited") and not re.search(
                        r"\b(was|were|is|are|we)\b", sentence[:match.start()], re.I):
                    continue
                if re.search(r"\b(these|those|this|their|them|its|it|most)\b", sentence, re.I):
                    continue
                opposite = sentence[:match.start()] + OPPOSITES[match.group()] + sentence[match.end():]
                prompts = [make_prompt(sentence, claim) for claim in (sentence, opposite)]
                if max(len(p.encode()) for p in prompts) > MAX_PROMPT:
                    continue
                candidates.append({**source, "evidence": sentence, "claim": sentence,
                                   "opposite_claim": opposite, "effect": match.group(),
                                   "element_path": f"front/article-meta/abstract[{ai + 1}]//p[{pi + 1}]",
                                   "sentence_index": si,
                                   "label_provenance": "AI-assisted constructed excerpt-agreement contrast, not human-reviewed",
                                   "modification": "Original complete sentence retained. One direction word in the query changed to its opposite. Answers constructed."})
    return candidates, "eligible" if candidates else "no_compact_direction_sentence"


def make_prompt(evidence, claim):
    return f"Does the claim agree with the excerpt?\nExcerpt: {evidence}\nClaim: {claim}"


def make_answer(claim, verdict):
    return f"{verdict}. The claim {'agrees' if verdict == 'Yes' else 'disagrees'} with the excerpt: {claim}"


def pairs(original, group):
    rows = []
    for claim, verdict, variant in ((original["claim"], "Yes", "original_claim"),
                                    (original["opposite_claim"], "No", "opposite_direction")):
        rows.append({**original, "group_id": group, "variant": variant,
                     "prompt": make_prompt(original["evidence"], claim),
                     "chosen": make_answer(claim, verdict), "rejected": make_answer(claim, "No" if verdict == "Yes" else "Yes"),
                     "margin": 0.0, "label_kind": "preference_pair", "human_reviewed": False})
    return rows


def capture(out, count):
    if not 1 <= count <= 8192:
        raise ValueError("bounded inventory sample count required")
    out.mkdir(parents=True, exist_ok=True)
    raw, receipt = read_url("https://api.github.com/repos/elifesciences/elife-article-xml/git/trees/" + TREE + "?recursive=1", 16 << 20)
    listing = json.loads(raw)
    if listing.get("truncated") is not False or listing.get("sha") != TREE:
        raise ValueError("complete pinned article inventory required")
    (out / "article-tree.json").write_bytes(raw)
    write_json(out / "inventory-receipt.json", receipt)
    files = [e for e in listing["tree"] if e.get("type") == "blob" and re.fullmatch(r"elife-\d+-v1\.xml", e["path"])]
    ordered = sorted(files, key=lambda e: sha(("RESEARCH_QUERY-source-v1:" + e["path"]).encode()))[:count]
    if len(files) != 19699 or len(ordered) != count:
        raise ValueError("pinned v1 inventory changed")
    write_json(out / "selection.json", {"inventory_entries": len(listing["tree"]), "v1_files": len(files),
               "truncated": False, "inspected_sample": count, "selection": "SHA-256 order across complete pinned v1 inventory, not first page",
               "files": ordered, "scope": "journal research article abstracts only; not reviewed preprints"})
    results, failures = [], []
    def one(entry):
        path = out / "xml" / entry["path"]
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            payload = path.read_bytes()
            rec = {"url": RAW + entry["path"], "sha256": sha(payload), "bytes": len(payload), "cached": True}
        else:
            payload, rec = read_url(RAW + entry["path"], MAX_XML)
            path.write_bytes(payload)
        rows, disposition = sentence_candidates(payload, entry["path"], entry["sha"])
        return {"file": entry["path"], "receipt": rec, "disposition": disposition, "candidates": rows}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        work = {executor.submit(one, e): e for e in ordered}
        for future in concurrent.futures.as_completed(work):
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"file": work[future]["path"], "error": type(exc).__name__ + ": " + str(exc)})
            if (len(results) + len(failures)) % 64 == 0:
                write_json(out / "capture-progress.json", {"complete": len(results), "failures": failures, "requested": count})
    results.sort(key=lambda e: e["file"])
    write_json(out / "source-results.json", {"schema_version": 1, "revision": REVISION, "results": results, "failures": failures,
               "counts": dict(Counter(e["disposition"] for e in results)), "sample_size": count, "entire_inventory_xml_read": False})
    return results, failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", type=int, default=512)
    args = parser.parse_args()
    results, failures = capture(args.out, args.count)
    print(json.dumps({"files_read": len(results), "failed": len(failures), "dispositions": dict(Counter(e["disposition"] for e in results)),
                      "candidate_sentences": sum(len(e["candidates"]) for e in results)}))
    raise SystemExit(1 if failures else 0)
