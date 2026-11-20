# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Full anonymous primary inventory, not a first-page population estimate."""
import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from data import HERE, digest, write_json

PATTERN = ("?entity p:P36 ?statement. "
           "?statement ps:P36 ?capital; prov:wasDerivedFrom ?reference.")
QUERIES = {
    "count": "SELECT (COUNT(DISTINCT ?entity) AS ?count) WHERE { " + PATTERN + " }",
    "entities": "SELECT DISTINCT ?entity WHERE { " + PATTERN + " } ORDER BY STR(?entity) LIMIT 10000",
}
MAX_BYTES = 4 << 20
MAX_ENTITIES = 100000
SPACING_SECONDS = 65


def page_query(cursor):
    if cursor is None:
        return QUERIES["entities"]
    if not re.fullmatch(r"Q[1-9][0-9]*", cursor):
        raise ValueError("invalid inventory keyset cursor")
    return ("SELECT DISTINCT ?entity WHERE { " + PATTERN
            + ' FILTER(STR(?entity) > "http://www.wikidata.org/entity/' + cursor
            + '") } ORDER BY STR(?entity) LIMIT 10000')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch(query):
    if query not in QUERIES.values():
        cursor = re.search(r'/entity/(Q[1-9][0-9]*)"', query)
        if cursor is None or query != page_query(cursor.group(1)):
            raise ValueError("only exact aggregate and keyset inventory queries are permitted")
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query, "format": "json"})
    request = urllib.request.Request(url, headers={
        "User-Agent": "curl/8.0 FactcheckInventory/2 (bounded anonymous research)",
        "Accept": "application/sparql-results+json",
    })
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=45) as response:
        raw = response.read(MAX_BYTES + 1)
        if response.status != 200 or len(raw) > MAX_BYTES:
            raise ValueError("primary inventory status or byte ceiling")
    json.loads(raw)
    return raw, url


def validate(count_raw, entities_raw):
    counts = json.loads(count_raw)["results"]["bindings"]
    if len(counts) != 1 or not re.fullmatch(r"[0-9]+", counts[0]["count"]["value"]):
        raise ValueError("primary aggregate must contain one integer count")
    count = int(counts[0]["count"]["value"])
    if not 1 <= count <= MAX_ENTITIES:
        raise ValueError("complete population outside the bounded inventory capacity")
    values = json.loads(entities_raw)["results"]["bindings"]
    ids = []
    for row in values:
        entity = row["entity"]
        value = entity["value"]
        if entity["type"] != "uri" or not re.fullmatch(r"http://www.wikidata.org/entity/Q[1-9][0-9]*", value):
            raise ValueError("non-primary inventory identity")
        ids.append(value.rsplit("/", 1)[1])
    if len(ids) != count or len(set(ids)) != count:
        raise ValueError("full inventory disagrees with the independent primary aggregate")
    return ids


def collect(root):
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {
        "schema_version": 1, "policy": "full-keyset-v3", "queries": QUERIES,
        "records": [], "attempts": 0, "complete": False,
    }
    if index["queries"] != QUERIES:
        raise ValueError("inventory query policy changed across resume")
    # Keep unmodified CC0 source responses beside their original granting text.
    (root / "SOURCE-CC0.txt").write_bytes((HERE / "SOURCE-CC0.txt").read_bytes())
    def get(name, query):
        filename = name + ".json"
        record = next((item for item in index["records"] if item["file"] == filename), None)
        if record:
            raw = (root / filename).read_bytes()
            expected_url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query, "format": "json"})
            if len(raw) > MAX_BYTES or digest(raw) != record["sha256"] or record["url"] != expected_url:
                raise ValueError("cached primary inventory identity changed")
        else:
            if index["attempts"] >= 16:
                raise ValueError("bounded inventory request budget exhausted")
            previous = index.get("last_request_at", 0)
            time.sleep(max(0, previous + SPACING_SECONDS - time.time()))
            index["attempts"] += 1
            index["last_request_at"] = time.time()
            write_json(index_path, index)
            try:
                raw, url = fetch(query)
            except Exception as error:
                refusal = {"error": type(error).__name__ + ": " + str(error),
                           "query": query, "attempt": index["attempts"]}
                index.setdefault("refusals", []).append(refusal)
                write_json(index_path, index)
                write_json(root / "refusal.json", refusal)
                if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                    # The next request waits on the same durable 65-second clock.
                    # Every refusal consumes the fixed budget. Never change hosts
                    # or identities to bypass a primary-source rate limit.
                    return get(name, query)
                raise
            (root / filename).write_bytes(raw)
            index["records"].append({"file": filename, "url": url, "bytes": len(raw),
                                     "sha256": digest(raw), "at": datetime.now(timezone.utc).isoformat()})
            write_json(index_path, index)
        return raw
    count_raw = get("count", QUERIES["count"])
    count = int(json.loads(count_raw)["results"]["bindings"][0]["count"]["value"])
    if not 1 <= count <= MAX_ENTITIES:
        raise ValueError("primary population exceeds complete inventory capacity")
    bindings, cursor = [], None
    for page in range(11):
        raw = get("page-" + str(page).zfill(2), page_query(cursor))
        values = json.loads(raw)["results"]["bindings"]
        if not values or len(values) > 10000:
            raise ValueError("premature empty or oversized keyset page")
        urls = [row["entity"]["value"] for row in values]
        if urls != sorted(set(urls)) or cursor and urls[0] <= "http://www.wikidata.org/entity/" + cursor:
            raise ValueError("nonmonotonic or duplicate keyset inventory")
        bindings.extend(values)
        if len(bindings) >= count:
            break
        cursor = urls[-1].rsplit("/", 1)[1]
        print("Inventory rows observed", len(bindings), "of aggregate", count, flush=True)
    combined = json.dumps({"results": {"bindings": bindings}}, separators=(",", ":")).encode()
    if len(combined) > 32 << 20:
        raise ValueError("combined primary inventory byte ceiling")
    ids = validate(count_raw, combined)
    final_count_raw = get("final-count", QUERIES["count"])
    validate(final_count_raw, combined)
    (root / "entities.json").write_bytes(combined)
    old = json.loads((HERE / "sources/discovery.json").read_text())["results"]["bindings"]
    old_ids = {row["entity"]["value"].rsplit("/", 1)[1] for row in old}
    index.update({"complete": True, "population": len(ids), "prior_pilot": len(old_ids),
                  "outside_prior_pilot": len(set(ids) - old_ids), "offsets_used": False,
                  "page_limit": 10000, "complete_count_verified_before_and_after": True,
                  "entities_sha256": digest(combined), "atomic_snapshot": False,
                  "population_is_source_eligible_claim": False,
                  "scope": "All currently indexed referenced P36 subjects, not validated statement eligibility",
                  "source_grant": "CC0-1.0", "source_grant_sha256": digest((root / "SOURCE-CC0.txt").read_bytes())})
    write_json(index_path, index)
    print(json.dumps(index, indent=2))
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    collect(parser.parse_args().out)
