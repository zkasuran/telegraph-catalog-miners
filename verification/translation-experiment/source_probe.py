# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Probe primary translation lineage only. Never load a model or make labels."""
import argparse
import bz2
import hashlib
import json
import tarfile
import time
import unicodedata
import urllib.request
from array import array
from collections import Counter
from pathlib import Path

SENTENCES = {
    "eng": "c2e2edd063a31cbbc8e416a4858225e934bbb0afff485031821e59f230ccf1ca",
    "fra": "cf4509f2f3fd542222be2cf085d0d814ce3f9405263c5746bcc965d7e83dfd08",
}
EXPORT = "https://downloads.tatoeba.org/exports/"
MAX_DOWNLOAD = 160 << 20
MAX_DECOMPRESSED = 1 << 30
MAX_RECORD = 8192
MAX_ROWS = 40000000
MAX_ID = 20000000
LINKS_SHA = "08a802bfee9242fe2476836ec94e180588149f047451bce7519331b871b19e4f"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def fetch(url, path, ceiling=MAX_DOWNLOAD):
    started = time.monotonic()
    request = urllib.request.Request(url, headers={"User-Agent": "Translation-source-probe/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response, path.open("wb") as out:
        if response.status != 200 or int(response.headers.get("Content-Length", 0)) > ceiling:
            raise ValueError("unexpected HTTP status or declared download size")
        total = 0
        while chunk := response.read(1 << 20):
            total += len(chunk)
            if total > ceiling or time.monotonic() - started > 240:
                raise ValueError("download byte or wall ceiling exceeded")
            out.write(chunk)
        if response.headers.get("Content-Length") and total != int(response.headers["Content-Length"]):
            raise ValueError("partial download")
        return {"url": url, "bytes": total, "sha256": sha(path),
                "last_modified": response.headers.get("Last-Modified"),
                "content_type": response.headers.get("Content-Type")}


def records(stream, fields):
    total, rows = 0, 0
    started = time.monotonic()
    while raw := stream.readline(MAX_RECORD + 1):
        total += len(raw)
        rows += 1
        if len(raw) > MAX_RECORD or total > MAX_DECOMPRESSED or rows > MAX_ROWS or time.monotonic() - started > 240:
            raise ValueError("decompressed record, count, byte or wall ceiling exceeded")
        # LF alone separates records. Do not strip or normalize original text.
        line = raw.removesuffix(b"\n").removesuffix(b"\r")
        parts = line.decode("utf-8").split("\t")
        if len(parts) != fields:
            raise ValueError("unexpected TSV schema")
        yield rows, parts


def sentences(path, language):
    if sha(path) != SENTENCES[language]:
        raise ValueError("CC0 source hash mismatch: " + language)
    result = {}
    with bz2.open(path, "rb") as stream:
        for number, parts in records(stream, 4):
            sid, lang, text, modified = parts
            if not sid.isascii() or not sid.isdigit() or int(sid) <= 0 or lang != language or not text or int(sid) in result:
                raise ValueError("invalid or duplicate original sentence identity")
            result[int(sid)] = {"id": int(sid), "language": lang, "text": text,
                                "modified": modified, "line": number, "license": "CC0-1.0"}
    return result


def link_join(path, originals):
    edges, touching, count = set(), set(), 0
    with tarfile.open(path, "r|bz2") as archive:
        member = next(iter(archive))
        if member.name != "links.csv" or not member.isfile() or member.size > MAX_DECOMPRESSED:
            raise ValueError("unexpected link member")
        with archive.extractfile(member) as stream:
            for count, (left, right) in records(stream, 2):
                if not left.isascii() or not right.isascii() or not left.isdigit() or not right.isdigit():
                    raise ValueError("invalid link identity")
                left, right = int(left), int(right)
                if left <= 0 or right <= 0 or left == right:
                    raise ValueError("nonpositive or self link")
                if left in originals or right in originals:
                    touching.add((left, right))
                if left in originals and right in originals:
                    if (left, right) in edges:
                        raise ValueError("duplicate selected link")
                    edges.add((left, right))
        if archive.next() is not None:
            raise ValueError("extra link archive member")
    missing = sorted((a, b) for a, b in edges if (b, a) not in edges)
    pairs = sorted((a, b) for a, b in edges if originals[a]["language"] == "eng" and originals[b]["language"] == "fra")
    return {"all_directed_rows": count, "selected_directed_edges": len(edges),
            "eng_fra_pairs": len(pairs), "missing_reciprocals": missing,
            "touching_directed_edges": len(touching)}, edges, pairs, touching


def components(path, originals, pairs, id_ceiling=MAX_ID):
    """Union the complete link graph, including paths through unselected languages."""
    parent = array("I", range(id_ceiling + 1))
    rank = bytearray(id_ceiling + 1)

    def root(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid

    def union(a, b):
        a, b = root(a), root(b)
        if a == b:
            return
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

    maximum = 0
    with tarfile.open(path, "r|bz2") as archive:
        member = next(iter(archive))
        with archive.extractfile(member) as stream:
            for _, (left, right) in records(stream, 2):
                a, b = int(left), int(right)
                maximum = max(maximum, a, b)
                if not 0 < a <= id_ceiling or not 0 < b <= id_ceiling:
                    raise ValueError("link graph ID ceiling exceeded")
                union(a, b)
    normalized = {}
    for sid, row in originals.items():
        key = (row["language"], norm(row["text"]))
        if key in normalized:
            union(sid, normalized[key])
        normalized[key] = sid
    wanted = {root(a) for a, _ in pairs}
    members = {r: [] for r in wanted}
    for sid in originals:
        r = root(sid)
        if r in members:
            members[r].append(sid)
    groups = []
    for r, ids in members.items():
        ids.sort()
        groups.append({"id": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                       "cc0_original_ids": ids,
                       "english_french_direct_links": sum(root(a) == r for a, _ in pairs)})
    return {"graph_scope": "All directed export links, including every language and non-CC0 ID; only IDs read",
            "maximum_sentence_id": maximum, "normalized_languages": ["eng", "fra"],
            "independent_link_components": len(groups), "groups": sorted(groups, key=lambda g: g["id"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"family": "LANGUAGE_TRANSLATION", "state": "PROBE_ONLY", "model_execution": False,
              "preference_rows": 0, "sources": {}, "blockers": []}
    try:
        originals = {}
        for lang in SENTENCES:
            path = args.source_cache / (lang + "-cc0.tsv.bz2")
            source = sentences(path, lang)
            originals.update(source)
            report["sources"][lang] = {"url": EXPORT + f"per_language/{lang}/{lang}_sentences_CC0.tsv.bz2",
                "sha256": sha(path), "rows": len(source), "license": "CC0-1.0",
                "short_rows_8_256_bytes": sum(8 <= len(r["text"].encode()) <= 256 for r in source.values()),
                "normalized_duplicate_count": len(source) - len({norm(r["text"]) for r in source.values()})}
        for name, url in {
            "downloads.html": "https://tatoeba.org/en/downloads",
            "terms.html": "https://tatoeba.org/en/terms_of_use",
            "faq.html": "https://en.wiki.tatoeba.org/articles/show/faq",
            "cc0.html": "https://creativecommons.org/publicdomain/zero/1.0/legalcode",
            "cc-by-2-fr.html": "https://creativecommons.org/licenses/by/2.0/fr/legalcode",
        }.items():
            report["sources"][name] = fetch(url, args.out / name, 2 << 20)
        links = args.out / "links.tar.bz2"
        if links.is_file() and sha(links) == LINKS_SHA:
            report["sources"]["links"] = {"url": EXPORT + "links.tar.bz2", "sha256": LINKS_SHA,
                "bytes": links.stat().st_size, "last_modified": "Sat, 14 Nov 2026 06:56:19 GMT",
                "content_type": "application/octet-stream", "hash_verified_cached_capture": True}
        else:
            report["sources"]["links"] = fetch(EXPORT + "links.tar.bz2", links)
        if report["sources"]["links"]["sha256"] != LINKS_SHA:
            raise ValueError("link snapshot hash mismatch")
        audit, edges, pairs, touching = link_join(links, originals)
        report["links"] = audit
        if not pairs or audit["missing_reciprocals"]:
            raise ValueError("No linked originals or selected directed links lack reciprocal identities")
        # IDs only. This is source lineage, not an approved preference corpus.
        write(args.out / "link-identities.json", {
            "source_sha256": report["sources"]["links"]["sha256"],
            "license": "CC-BY-2.0-FR", "attribution": "Tatoeba and contributors, https://tatoeba.org",
            "eng_fra_original_ids": pairs, "selected_edges": sorted(edges),
            "touching_edges": sorted(touching),
        })
        graph = components(links, originals, pairs)
        write(args.out / "components.json", graph)
        report["components"] = {k: v for k, v in graph.items() if k != "groups"}
        report["pilot_source_group_requirement"] = {"train": 128, "validation": 32, "sealed_final": 64}
        report["blockers"].append(
            f"Only {graph['independent_link_components']} independent English/French link components; "
            "the recorded 128/32/64 source-group pilot requires 224 before donor pairing")
        report["original_identity_digest"] = hashlib.sha256(json.dumps(originals, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        report["language_counts"] = dict(Counter(r["language"] for r in originals.values()))
        report["rights"] = {"sentence_license": "CC0-1.0", "link_license": "CC-BY-2.0-FR",
                           "blanket_sand": False, "attribution": "Tatoeba and contributors, https://tatoeba.org",
                           "link_grant_source": "downloads.html general Creative Commons section",
                           "source_rights_survive_failed_checkpoints": True}
        report["blockers"].append("No bilingual-reviewed negative preferences or sealed final audit; no dataset or workflow enabled")
    except (OSError, ValueError, tarfile.TarError) as error:
        report["state"] = "SOURCE_PROBE_FAILED"
        report["blockers"].append(str(error))
    write(args.out / "source-probe.json", report)
    print(json.dumps(report, indent=2))
    return 1 if report["state"] == "SOURCE_PROBE_FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
