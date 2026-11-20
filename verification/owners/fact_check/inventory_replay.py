# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Source-only full-inventory replay. No live query or neural import."""
import argparse
import io
import json
import tarfile
from pathlib import Path

from data import HERE, digest, write_json
from inventory import QUERIES, page_query, validate

MAX_ARCHIVE = 4 << 20
MAX_EXPANDED = 32 << 20


def decode_archive(raw, manifest):
    if len(raw) > MAX_ARCHIVE or digest(raw) != manifest["archive_sha256"]:
        raise ValueError("full source inventory archive hash or byte ceiling")
    expected = {item["file"]: item for item in manifest["files"]}
    if len(expected) != len(manifest["files"]) or not 1 <= len(expected) <= 32:
        raise ValueError("inventory archive must have a bounded unique file manifest")
    result = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or sum(member.size for member in members) > MAX_EXPANDED:
            raise ValueError("inventory archive file count or expansion ceiling")
        for member in members:
            if (not member.isfile() or Path(member.name).name != member.name or member.name not in expected
                    or member.name in result or member.size > 4 << 20):
                raise ValueError("inventory archive path, type or duplicate entry")
            body = archive.extractfile(member).read()
            if len(body) != expected[member.name]["bytes"] or digest(body) != expected[member.name]["sha256"]:
                raise ValueError("primary inventory entry identity mismatch")
            result[member.name] = body
    return result


def verify_sources(files, manifest):
    original = json.loads(files["checkpoint-index.json"])
    complete = json.loads(files["completion.json"])
    if (original["complete"] is not False or original["attempts"] != 16
            or complete["complete"] is not True or complete["checkpoint_sha256"] != digest(files["checkpoint-index.json"])
            or original["queries"] != QUERIES or complete["original_attempts"] != original["attempts"]
            or not 1 <= complete["completion_attempts"] <= 8
            or complete["completion_request_limit"] != 8
            or original["attempts"] != len(original["records"]) + len(original["refusals"])
            or complete["completion_attempts"] != len(complete["records"]) + len(complete["refusals"])
            or complete["total_transport_attempts"] != 16 + complete["completion_attempts"]):
        raise ValueError("inventory completion resets history or lacks a full verified boundary")
    if (complete["original_source_run_id"] != manifest["source_run_id"]
            or complete["original_source_commit"] != manifest["source_commit"]):
        raise ValueError("full inventory source run/commit binding mismatch")
    records = original["records"] + complete["records"]
    if not {record["file"] for record in records} <= set(files):
        raise ValueError("full inventory is missing required primary records")
    if len({record["file"] for record in records}) != len(records):
        raise ValueError("repeated primary inventory record")
    pages = [record for record in records if record["file"].startswith("page-")]
    if [record["file"] for record in pages] != ["page-" + str(i).zfill(2) + ".json" for i in range(len(pages))]:
        raise ValueError("full inventory has missing or reordered pages")
    if not 1 <= len(pages) <= 10 or {record["file"] for record in records} != {"count.json", "final-count.json", *(record["file"] for record in pages)}:
        raise ValueError("full inventory requires all pages and both aggregate counts")
    values, cursor = [], None
    import urllib.parse
    for record in records:
        body = files[record["file"]]
        if digest(body) != record["sha256"] or len(body) != record["bytes"]:
            raise ValueError("original primary record bytes changed")
        query = page_query(cursor) if record["file"].startswith("page-") else QUERIES["count"]
        url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query, "format": "json"})
        if record["url"] != url:
            raise ValueError("primary inventory query/cursor identity differs")
        if not record["file"].startswith("page-"):
            continue
        rows = json.loads(body)["results"]["bindings"]
        urls = [row["entity"]["value"] for row in rows]
        if not 1 <= len(rows) <= 10000 or urls != sorted(set(urls)) or cursor and urls[0] <= "http://www.wikidata.org/entity/" + cursor:
            raise ValueError("full inventory pages are not unique and monotonic")
        values.extend(rows)
        cursor = urls[-1].rsplit("/", 1)[1]
    combined = json.dumps({"results": {"bindings": values}}, separators=(",", ":")).encode()
    ids = validate(files["count.json"], combined)
    validate(files["final-count.json"], combined)
    if len(ids) != manifest["population"] or len(ids) != complete["population"] or digest(combined) != complete["entities_sha256"]:
        raise ValueError("full inventory recount or normalized payload hash changed")
    if digest(files["SOURCE-CC0.txt"]) != "a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499":
        raise ValueError("primary source grant bytes changed")
    return {"status": "FULL_SOURCE_INVENTORY_PASS", "family": "FACT_CHECK", "population": len(ids),
            "page_count": len(pages), "normalized_id_duplicates": 0, "aggregate_count_verified_before_and_after": True,
            "source_manifest_sha256": digest((HERE / "inventory-manifest.json").read_bytes()),
            "archive_sha256": manifest["archive_sha256"], "source_run_id": manifest["source_run_id"],
            "source_commit": manifest["source_commit"], "atomic_snapshot": False,
            "source_eligible_claims_verified": False, "model_or_audit_scored": False, "network_requests": 0}


def replay(root=None):
    manifest = json.loads((HERE / "inventory-manifest.json").read_text())
    archive = HERE / "inventory-snapshot.tar.gz"
    if archive.is_symlink() or archive.stat().st_size > MAX_ARCHIVE:
        raise ValueError("inventory must be a bounded regular archive")
    files = decode_archive(archive.read_bytes(), manifest)
    report = verify_sources(files, manifest)
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "inventory-gate.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    replay(parser.parse_args().out)
