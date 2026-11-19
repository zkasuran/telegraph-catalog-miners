# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Bounded public annotation/metadata capture. No models or private pool reads."""
import argparse
import gzip
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSION = "https://api.figshare.com/v2/articles/7722905/versions/1"
FILE = "https://ndownloader.figshare.com/files/14376173"
MD5 = "64e93f32da691bc756a25a3fa8e44451"
SELECT = "id,title,doi,ids,publication_year"
MAX_JSON = 8 << 20
MAX_EXPANDED = 32 << 20


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode()
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("refusing to replace existing capture: " + str(path))
    if not path.exists():
        path.write_bytes(raw)


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        host = urllib.parse.urlsplit(newurl).hostname or ""
        docs_redirect = urllib.parse.urlsplit(req.full_url).hostname in {"docs.openalex.org", "help.openalex.org"} and host == "help.openalex.org"
        figshare_storage = urllib.parse.urlsplit(req.full_url).hostname == "ndownloader.figshare.com" and host == "44c1cec316ec56bca76dfd65d2882b20.r2.cloudflarestorage.com"
        if not newurl.startswith("https://") or not (docs_redirect or figshare_storage or host.endswith(".amazonaws.com") or host.endswith(".figshare.com")):
            raise ValueError("unapproved source redirect to " + urllib.parse.urlsplit(newurl).scheme + "://" + host)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url, path, ceiling=MAX_JSON):
    if path.exists():
        if path.is_symlink() or path.stat().st_size > ceiling:
            raise ValueError("cached source is not bounded regular bytes")
        raw = path.read_bytes()
        receipt = json.loads(path.with_suffix(path.suffix + ".receipt.json").read_bytes())
        if receipt["requested_url"] != url or receipt["bytes"] != len(raw) or receipt["sha256"] != sha(raw):
            raise ValueError("cached source receipt identity mismatch")
        return raw
    host = urllib.parse.urlsplit(url).hostname
    if host not in {"api.figshare.com", "ndownloader.figshare.com", "api.openalex.org", "help.openalex.org", "creativecommons.org", "docs.openalex.org"}:
        raise ValueError("source host not permitted")
    request = urllib.request.Request(url, headers={"User-Agent": "academic-title-source-audit/1"})
    start = time.monotonic()
    with urllib.request.build_opener(SafeRedirect()).open(request, timeout=30) as response:
        if response.status != 200:
            raise ValueError("source HTTP status")
        raw = response.read(ceiling + 1)
        # Do not retain a signed storage redirect URL or query parameters.
        receipt = {"requested_url": url, "at": datetime.now(timezone.utc).isoformat(),
                   "bytes": len(raw), "sha256": sha(raw), "http_status": response.status,
                   "content_type": response.headers.get("Content-Type"), "elapsed_seconds": time.monotonic() - start}
    if len(raw) > ceiling or receipt["elapsed_seconds"] > 90:
        raise ValueError("source byte/time ceiling")
    path.write_bytes(raw)
    write(path.with_suffix(path.suffix + ".receipt.json"), receipt)
    return raw


def annotations(raw):
    if hashlib.md5(raw).hexdigest() != MD5:
        raise ValueError("RELISH published checksum mismatch")
    with gzip.GzipFile(fileobj=__import__("io").BytesIO(raw)) as archive:
        expanded = archive.read(MAX_EXPANDED + 1)
    if len(expanded) > MAX_EXPANDED:
        raise ValueError("RELISH expansion ceiling")
    values = json.loads(expanded)
    if not isinstance(values, list) or len(values) > 10000:
        raise ValueError("RELISH top-level shape/count")
    out = []
    for row in values:
        if not isinstance(row, dict) or not {"pmid", "uid", "response"} <= set(row):
            raise ValueError("RELISH missing source fields")
        seed = str(row["pmid"])
        if not re.fullmatch(r"[1-9][0-9]{0,8}", seed):
            raise ValueError("RELISH invalid seed PMID")
        response = row["response"]
        if not isinstance(response, dict) or set(response) != {"relevant", "partial", "irrelevant"}:
            raise ValueError("RELISH judgment schema")
        projected = {"uid": str(row["uid"]), "pmid": seed}
        for label, ids in response.items():
            if not isinstance(ids, list) or len(ids) > 1000:
                raise ValueError("RELISH judgment list ceiling")
            projected[label] = sorted({str(i) for i in ids})
            if any(re.fullmatch(r"[1-9][0-9]{0,8}", i) is None for i in projected[label]):
                raise ValueError("RELISH invalid candidate PMID")
        out.append(projected)
    return out


def consensus(rows):
    labels = defaultdict(set)
    uids = defaultdict(set)
    for row in rows:
        for label in ("relevant", "partial", "irrelevant"):
            for ident in row[label]:
                labels[row["pmid"], ident].add(label)
                uids[row["pmid"], ident].add(row["uid"])
    selected = defaultdict(lambda: {"relevant": [], "irrelevant": []})
    for (seed, target), votes in sorted(labels.items()):
        if seed != target and len(votes) == 1 and next(iter(votes)) in ("relevant", "irrelevant"):
            label = next(iter(votes))
            selected[seed][label].append({"pmid": target, "annotation_uids": sorted(uids[seed, target])})
    return dict(selected), {"source_rows": len(rows), "unique_seed_pmids": len({r["pmid"] for r in rows}),
                            "unique_seed_candidate_pairs": len(labels),
                            "conflicting_pairs_excluded": sum(len(v) > 1 for v in labels.values()),
                            "source_label_pair_counts": dict(Counter(next(iter(v)) for v in labels.values() if len(v) == 1)),
                            "seeds_with_relevant_and_irrelevant": sum(bool(v["relevant"] and v["irrelevant"]) for v in selected.values())}


def validate_batch(body, requested):
    if not isinstance(body, dict) or not isinstance(body.get("results"), list):
        raise ValueError("OpenAlex batch schema")
    if body.get("meta", {}).get("count") != len(body["results"]) or len(body["results"]) > 100:
        raise ValueError("OpenAlex page is not complete")
    found = {}
    ambiguous = set()
    for row in body["results"]:
        if not isinstance(row, dict) or set(row) != set(SELECT.split(",")) or not isinstance(row.get("ids"), dict):
            raise ValueError("unrequested metadata fields returned")
        ident = str(row["ids"].get("pmid", "")).rstrip("/").rsplit("/", 1)[-1]
        if ident not in requested or row["ids"].get("openalex") != row["id"]:
            raise ValueError("OpenAlex returned another identity")
        if ident in found:
            ambiguous.add(ident)
        found[ident] = row
    for ident in ambiguous:
        del found[ident]
    return found, sorted(set(requested) - set(found)), sorted(ambiguous)


def capture(seed_limit, donors):
    if not 1 <= seed_limit <= 1500 or not 1 <= donors <= 8:
        raise ValueError("pilot source request ceiling")
    root = HERE / "sources"
    root.mkdir(exist_ok=True)
    version = json.loads(fetch(VERSION, root / "relish-version.json"))
    if version["license"]["name"] != "CC0" or version["version"] != 1:
        raise ValueError("RELISH primary source grant changed")
    files = version["files"]
    if len(files) != 1 or files[0]["id"] != 14376173 or files[0]["computed_md5"] != MD5 or files[0]["size"] != 978490:
        raise ValueError("RELISH primary file inventory mismatch")
    project = []
    for page in (1, 2):
        data = json.loads(fetch("https://api.figshare.com/v2/projects/60095/articles?page_size=100&page=" + str(page), root / ("project-page-" + str(page) + ".json")))
        project.extend(data)
        if page == 2 and data:
            raise ValueError("RELISH project listing needs further pages")
    if [r["id"] for r in project] != [7722905]:
        raise ValueError("RELISH relevant source inventory changed")
    fetch("https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt", root / "CC0.txt")
    fetch("https://docs.openalex.org/how-to-use-the-api/api-overview", root / "openalex-api.html")
    fetch("https://help.openalex.org/api/filtering/", root / "openalex-filtering.html")
    rows = annotations(fetch(FILE, root / "RELISH_v1.json.gz", 2 << 20))
    pool, report = consensus(rows)
    seeds = sorted((s for s, v in pool.items() if v["relevant"] and v["irrelevant"]), key=lambda s: sha(("RELISH-source-v1:" + s).encode()))[:seed_limit]
    wanted = set(seeds)
    for seed in seeds:
        for label in ("relevant", "irrelevant"):
            for row in sorted(pool[seed][label], key=lambda r: sha((seed + ":" + r["pmid"]).encode()))[:donors]:
                wanted.add(row["pmid"])
    wanted = sorted(wanted)
    metadata, missing, ambiguous = {}, [], []
    for offset in range(0, len(wanted), 100):
        ids = wanted[offset:offset + 100]
        params = urllib.parse.urlencode({"filter": "pmid:" + "|".join(ids), "select": SELECT, "per_page": 100})
        url = "https://api.openalex.org/works?" + params
        path = root / ("openalex-" + sha(url.encode())[:20] + ".json")
        body = json.loads(fetch(url, path))
        found, absent, multi = validate_batch(body, ids)
        metadata.update(found)
        missing.extend(absent)
        ambiguous.extend(multi)
        print("metadata batch", offset // 100 + 1, "bound", len(found), "of", len(ids), flush=True)
        time.sleep(0.2)
    report.update({"annotation_sha256": sha((root / "RELISH_v1.json.gz").read_bytes()),
                   "metadata_scope": "complete batches for hash-selected seeds and donors, not all OpenAlex",
                   "selected_seed_limit": seed_limit, "selected_seeds": seeds, "donors_per_label": donors,
                   "requested_metadata_ids": len(wanted), "bound_metadata_ids": len(metadata),
                   "missing_metadata_ids": sorted(set(missing)), "ambiguous_metadata_ids": sorted(set(ambiguous)),
                   "full_annotation_inventory_scanned": True, "project_listing_complete": True})
    write(HERE / "source-inventory.json", report)
    write(HERE / "metadata.json", metadata)
    write(HERE / "annotation-pool.json", pool)
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-limit", type=int, default=800)
    parser.add_argument("--donors-per-label", type=int, default=4)
    args = parser.parse_args()
    try:
        capture(args.seed_limit, args.donors_per_label)
    except Exception as exc:
        failure = {"at": datetime.now(timezone.utc).isoformat(), "phase": "source_capture",
                   "error_type": type(exc).__name__, "error": str(exc), "models_loaded": False}
        write(HERE / ("source-failure-" + str(time.time_ns()) + ".json"), failure)
        raise
