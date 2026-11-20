# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Anonymous bounded live class captures. Never reads the evaluation pool."""
import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from data import digest, write_json

FAMILY = "TEXT_SUMMARIZATION"
CHILD = FAMILY + "_GENERAL"
NODE = "https://devnode.telegraphprotocol.com"
URLS = {"family": NODE + "/api/wasm?resolve=" + FAMILY,
        "child": NODE + "/engine/v1/intents/" + CHILD}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def https_read(url):
    if url not in URLS.values():
        raise ValueError("class capture requires an exact primary endpoint")
    request = urllib.request.Request(url, headers={"User-Agent": "Summary-class-check/1", "Cache-Control": "no-cache"})
    start = time.monotonic()
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
        if response.status != 200 or response.geturl() != url or "json" not in response.headers.get("Content-Type", ""):
            raise ValueError("class source status, redirect or content type mismatch")
        raw = response.read((1 << 20) + 1)
    if len(raw) > 1 << 20 or time.monotonic() - start > 60:
        raise ValueError("class source byte or time ceiling")
    return raw.decode("utf-8")


def validate_snapshot(snapshot, now=None):
    if set(snapshot) != {"schema", "family", "child", "fetched_at"} or snapshot["schema"] != 1:
        raise ValueError("invalid live class snapshot envelope")
    current = now or datetime.now(timezone.utc)
    fetched = datetime.fromisoformat(snapshot["fetched_at"])
    if fetched.tzinfo is None or not -60 <= (current - fetched).total_seconds() <= 900:
        raise ValueError("class snapshot stale, future or missing timezone")
    for name, url in URLS.items():
        if snapshot[name]["url"] != url or set(snapshot[name]) != {"url", "raw"}:
            raise ValueError("primary source URL mismatch")
    parent = json.loads(snapshot["family"]["raw"])["resolve"]
    child = json.loads(snapshot["child"]["raw"])
    cls = parent.get("class")
    if (parent.get("intent") != FAMILY or cls != "judgment" or child.get("intent_id") != CHILD
            or child.get("intent_name") != CHILD or child.get("class") != cls
            or child.get("evaluation", {}).get("kind") != cls):
        raise ValueError("live family/child class agreement failed")
    # No response schema requirement is inferred. Any advertised schema
    # ownership still has to match this exact selected family.
    for field in ("request", "response", "onchain"):
        advertised = child.get(field)
        if isinstance(advertised, dict) and advertised.get("served_by") not in (None, "", FAMILY):
            raise ValueError("advertised summary schema belongs to another family")
    return cls


def capture(root):
    root.mkdir(parents=True, exist_ok=True)
    snapshot = {"schema": 1}
    for name, url in URLS.items():
        snapshot[name] = {"url": url, "raw": https_read(url)}
    snapshot["fetched_at"] = datetime.now(timezone.utc).isoformat()
    cls = validate_snapshot(snapshot)
    path = root / "class-snapshot.json"
    write_json(path, snapshot)
    result = {"class": cls, "family": FAMILY, "child": CHILD, "snapshot_sha256": digest(path.read_bytes()),
              "fetched_at": snapshot["fetched_at"], "source_sha256": {name: digest(snapshot[name]["raw"].encode()) for name in URLS},
              "urls": URLS, "pool_rows_read": False}
    write_json(root / "class-capture.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(capture(parser.parse_args().root), indent=2))
