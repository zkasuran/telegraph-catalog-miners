# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Complete an immutable partial inventory, then replay the full primary snapshot."""
import argparse
import json
import re
import shutil
import time
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from data import HERE, digest, write_json
from inventory import MAX_BYTES, QUERIES, SPACING_SECONDS, fetch, page_query, validate

COMPLETION_REQUESTS = 8


def load_checkpoint(root):
    raw = (root / "index.json").read_bytes()
    index = json.loads(raw)
    if index.get("complete") is not False or index.get("attempts") != 16 or index.get("queries") != QUERIES:
        raise ValueError("only the exact exhausted immutable source checkpoint may continue")
    records = index["records"]
    if index["attempts"] != len(records) + len(index.get("refusals", [])):
        raise ValueError("original transport history does not match its request count")
    pages = [record for record in records if record["file"].startswith("page-")]
    if [record["file"] for record in pages] != ["page-" + str(i).zfill(2) + ".json" for i in range(len(pages))]:
        raise ValueError("source checkpoint has a gap or reordered pages")
    if not 1 <= len(pages) <= 9 or len(records) != len(pages) + 1 or records[0]["file"] != "count.json":
        raise ValueError("source checkpoint must contain one count and contiguous complete pages")
    values, cursor = [], None
    for record in records:
        path = root / record["file"]
        if path.is_symlink() or path.parent != root or path.stat().st_size > MAX_BYTES:
            raise ValueError("source checkpoint path or byte ceiling")
        body = path.read_bytes()
        if digest(body) != record["sha256"]:
            raise ValueError("source checkpoint response identity changed")
        query = QUERIES["count"] if record["file"] == "count.json" else page_query(cursor)
        url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query, "format": "json"})
        if record["url"] != url:
            raise ValueError("source checkpoint query/cursor identity mismatch")
        if record["file"] == "count.json":
            continue
        rows = json.loads(body)["results"]["bindings"]
        urls = [row["entity"]["value"] for row in rows]
        if any(row["entity"]["type"] != "uri" or not re.fullmatch(r"http://www.wikidata.org/entity/Q[1-9][0-9]*", row["entity"]["value"]) for row in rows):
            raise ValueError("source checkpoint contains non-primary identities")
        if len(rows) != 10000 or urls != sorted(set(urls)) or cursor and urls[0] <= "http://www.wikidata.org/entity/" + cursor:
            raise ValueError("source checkpoint page is not complete and monotonic")
        values.extend(rows)
        cursor = urls[-1].rsplit("/", 1)[1]
    count_raw = (root / "count.json").read_bytes()
    count = int(json.loads(count_raw)["results"]["bindings"][0]["count"]["value"])
    if not 1 <= count - len(values) <= 10000:
        raise ValueError("completion must require only one bounded tail page")
    return index, values, cursor, count_raw, digest(raw)


def complete(checkpoint, output, binding):
    original, values, cursor, count_raw, checkpoint_sha = load_checkpoint(checkpoint)
    source_binding = json.loads(binding.read_text())
    if (source_binding["checkpoint_sha256"] != checkpoint_sha or source_binding["github_run_id"] != 44684179717
            or source_binding["commit"] != "ccf3bd21a80282a9b1165ab3b0b89e2051bec85d"):
        raise ValueError("continuation is not bound to the observed original run/checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    for record in original["records"]:
        target = output / record["file"]
        body = (checkpoint / record["file"]).read_bytes()
        if target.exists() and target.read_bytes() != body:
            raise ValueError("existing copied primary source changed")
        target.write_bytes(body)
    (output / "checkpoint-index.json").write_bytes((checkpoint / "index.json").read_bytes())
    for name in ("SOURCE-CC0.txt", "LICENSE", "NOTICE"):
        shutil.copyfile(HERE / name, output / name)
    path = output / "completion.json"
    state = json.loads(path.read_text()) if path.exists() else {
        "schema_version": 1, "family": "FACT_CHECK", "checkpoint_sha256": checkpoint_sha,
        "original_attempts": original["attempts"], "completion_attempts": 0, "completion_request_limit": COMPLETION_REQUESTS,
        "records": [], "refusals": [], "complete": False, "source_grant": "CC0-1.0",
        "original_source_run_id": 44684179717, "original_source_commit": "ccf3bd21a80282a9b1165ab3b0b89e2051bec85d",
    }
    if state["checkpoint_sha256"] != checkpoint_sha or state["original_attempts"] != 16:
        raise ValueError("continuation cannot replace or reset its original request history")
    def get(name, query):
        url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query, "format": "json"})
        cached = next((record for record in state["records"] if record["file"] == name), None)
        if cached:
            raw = (output / name).read_bytes()
            if cached["url"] != url or len(raw) > MAX_BYTES or digest(raw) != cached["sha256"]:
                raise ValueError("continuation cache identity changed")
            return raw
        while state["completion_attempts"] < COMPLETION_REQUESTS:
            previous = max(original.get("last_request_at", 0), state.get("last_request_at", 0))
            time.sleep(max(0, previous + SPACING_SECONDS - time.time()))
            state["completion_attempts"] += 1
            state["last_request_at"] = time.time()
            write_json(path, state)
            try:
                raw, received_url = fetch(query)
            except Exception as error:
                state["refusals"].append({"attempt": state["completion_attempts"], "error": type(error).__name__ + ": " + str(error), "query": query})
                write_json(path, state)
                if isinstance(error, TimeoutError) or isinstance(error, urllib.error.HTTPError) and error.code == 429:
                    continue
                raise
            if received_url != url:
                raise ValueError("continuation primary URL identity mismatch")
            (output / name).write_bytes(raw)
            state["records"].append({"file": name, "url": url, "sha256": digest(raw), "bytes": len(raw), "at": datetime.now(timezone.utc).isoformat()})
            write_json(path, state)
            return raw
        raise ValueError("bounded completion request budget exhausted")
    tail_name = "page-" + str(len(original["records"]) - 1).zfill(2) + ".json"
    tail = get(tail_name, page_query(cursor))
    tail_values = json.loads(tail)["results"]["bindings"]
    tail_urls = [row["entity"]["value"] for row in tail_values]
    if not tail_urls or len(tail_values) > 10000 or tail_urls != sorted(set(tail_urls)) or tail_urls[0] <= "http://www.wikidata.org/entity/" + cursor:
        raise ValueError("invalid or nonmonotonic continuation tail")
    values.extend(tail_values)
    combined = json.dumps({"results": {"bindings": values}}, separators=(",", ":")).encode()
    if len(combined) > 32 << 20:
        raise ValueError("full combined inventory byte ceiling")
    ids = validate(count_raw, combined)
    final_raw = get("final-count.json", QUERIES["count"])
    validate(final_raw, combined)
    (output / "entities.json").write_bytes(combined)
    state.update({"complete": True, "population": len(ids), "entities_sha256": digest(combined),
                  "total_transport_attempts": original["attempts"] + state["completion_attempts"],
                  "complete_count_verified_before_and_after": True, "atomic_snapshot": False,
                  "source_eligible_claims_verified": False, "original_checkpoint_unchanged": True})
    write_json(path, state)
    print(json.dumps(state, indent=2))
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    arguments = parser.parse_args()
    complete(arguments.checkpoint, arguments.out, arguments.binding)
