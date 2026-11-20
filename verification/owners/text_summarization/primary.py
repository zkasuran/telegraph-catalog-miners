# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Reproduce all selected annotations from the pinned primary XML bytes."""
import argparse
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from data import HERE, digest, fetch, norm, write_json

REVISION = "c3cd502f94edfb5aa0c5870ed9eff053773c6125"
PREFIX = "https://raw.githubusercontent.com/elifesciences/elife-article-xml/" + REVISION + "/articles/"
MAX_XML = 4 << 20
WORDS = re.compile(r"[a-zA-Z][a-zA-Z0-9-]{2,}")
STOP = {"the", "and", "with", "from", "for", "that", "this", "are", "was", "were", "its", "into", "through", "during", "between", "using", "their"}
GENERIC = {"new", "study", "studies", "research", "finding", "findings", "role", "roles", "reveals", "revealing", "insights", "understanding", "towards", "beyond"}


def anchors(own, other):
    def words(value):
        return set(WORDS.findall(value.casefold())) - STOP
    return sorted((words(own["title"]) & words(own["abstract"])) - words(other["abstract"]) - GENERIC)


def check_article(raw, document):
    filename = document["source_url"].removeprefix(PREFIX)
    if (not re.fullmatch(r"elife-[0-9]+-v1\.xml", filename) or len(raw) > MAX_XML
            or b"<!ENTITY" in raw or digest(raw) != document["source_sha256"]):
        raise ValueError("primary XML URL, entity or byte identity mismatch")
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw, usedforsecurity=False).hexdigest()
    if blob != document["git_blob"] or document["source_revision"] != REVISION:
        raise ValueError("primary Git source identity mismatch")
    root = ET.fromstring(raw)
    meta = root.find("front/article-meta")
    if meta is None:
        raise ValueError("primary article metadata missing")
    licenses = meta.findall("permissions/license")
    if (len(licenses) != 1 or licenses[0].get("{http://www.w3.org/1999/xlink}href", "").rstrip("/") not in
            ("http://creativecommons.org/licenses/by/4.0", "https://creativecommons.org/licenses/by/4.0")):
        raise ValueError("primary CC BY 4.0 grant missing")
    clause = " ".join("".join(licenses[0].itertext()).split())
    title = meta.find("title-group/article-title")
    abstracts = [a for a in meta.findall("abstract") if a.get("abstract-type") is None]
    if title is None or len(abstracts) != 1 or any(abstracts[0].find(".//" + tag) is not None for tag in ("xref", "ext-link", "blockquote")):
        raise ValueError("complete own abstract or title unavailable")
    paragraphs = [" ".join("".join(p.itertext()).split()) for p in abstracts[0].findall("p")]
    substantive = [p for p in paragraphs if not p.startswith("DOI:")]
    authors = [" ".join(" ".join("".join(n.itertext()) for n in c.findall("name/*")).split())
               for c in meta.findall("contrib-group/contrib") if c.get("contrib-type") == "author"]
    doi = next((x.text for x in meta.findall("article-id") if x.get("pub-id-type") == "doi"), None)
    extracted = {"doi": doi, "title": " ".join("".join(title.itertext()).split()), "abstract": "\n\n".join(substantive),
                 "authors": authors, "rights_clause": clause}
    if (any(document[k] != v for k, v in extracted.items()) or document["license"] != "CC-BY-4.0"
            or not authors or not substantive or "unrestricted use" not in clause or "credited" not in clause):
        raise ValueError("primary full text, annotation, author or grant mismatch")
    identities = {"doi:" + doi, *("name:" + norm(author) for author in authors)}
    for node in meta.findall("contrib-group/contrib/contrib-id"):
        if node.get("contrib-id-type", "").lower() == "orcid" and node.text:
            identities.add("orcid:" + node.text.strip().removeprefix("https://orcid.org/").removeprefix("http://orcid.org/"))
    dates = []
    for node in meta.findall("pub-date"):
        if node.get("pub-type") in ("epub", "ppub") or node.get("publication-format") in ("electronic", "print"):
            try:
                dates.append(date(int(node.findtext("year")), int(node.findtext("month")), int(node.findtext("day"))).isoformat())
            except (TypeError, ValueError):
                continue
    if not dates:
        raise ValueError("primary first-publication date missing")
    return identities, min(dates)


def verify_primary(root):
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / "data/manifest.json").read_text())
    groups = [g for split in manifest["splits"].values() for g in split]
    documents = [d for g in groups for d in g["documents"]]
    if len(documents) != 448:
        raise ValueError("primary source population changed")
    started = time.monotonic()

    def verify(document):
        if time.monotonic() - started > 1200 or not document["source_url"].startswith(PREFIX):
            raise ValueError("primary source deadline or revision")
        filename = document["source_url"].removeprefix(PREFIX)
        if not re.fullmatch(r"elife-[0-9]+-v1\.xml", filename):
            raise ValueError("unsafe primary article path")
        path = root / filename
        fetch(document["source_url"], path, document["source_sha256"], MAX_XML)
        return document["doi"], check_article(path.read_bytes(), document)

    with ThreadPoolExecutor(max_workers=6) as executor:
        checked = dict(executor.map(verify, documents))
    for group in groups:
        for d in group["documents"]:
            identities, published = checked[d["doi"]]
            if not identities <= set(group["source_entity_ids"]) or published > group["known_at"]:
                raise ValueError("actual primary author/ORCID/date group binding failed")
        for row in group["pairs"]:
            own, other = group["documents"] if row["target"] == "A" else reversed(group["documents"])
            if not anchors(own, other) or anchors(own, other) != row["exclusive_title_anchors"]:
                raise ValueError("exclusive title anchors are not primary-source-supported")
    report = {"grants_verified": True, "primary_documents": len(checked), "cropped": False,
              "source_manifest_sha256": digest((HERE / "data/manifest.json").read_bytes()),
              "source_revision": REVISION, "audit_scored": False, "elapsed_seconds": time.monotonic() - started}
    write_json(root.parent / "primary-checks.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(verify_primary(parser.parse_args().root)))
