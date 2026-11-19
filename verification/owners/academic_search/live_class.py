# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Fresh exact family/child class capture. No evaluation rows or credentials."""
import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from source import sha

FAMILY = "ACADEMIC_SEARCH"
CHILD = FAMILY + "_GENERAL"
URLS = {"family": "https://devnode.telegraphprotocol.com/api/wasm?resolve=" + FAMILY,
        "child": "https://devnode.telegraphprotocol.com/engine/v1/intents/" + CHILD}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("primary class source redirect refused")


def validate(snapshot, now=None):
    now = now or datetime.now(timezone.utc)
    if set(snapshot) != {"schema", "family", "child", "fetched_at"} or snapshot["schema"] != 1:
        raise ValueError("live class envelope mismatch")
    fetched = datetime.fromisoformat(snapshot["fetched_at"])
    if fetched.tzinfo is None or not -60 <= (now - fetched).total_seconds() <= 900:
        raise ValueError("stale or future class snapshot")
    for key, url in URLS.items():
        if set(snapshot[key]) != {"url", "raw"} or snapshot[key]["url"] != url:
            raise ValueError("exact primary class URL required")
    family = json.loads(snapshot["family"]["raw"])["resolve"]
    child = json.loads(snapshot["child"]["raw"])
    if (family.get("intent") != FAMILY or family.get("class") != "judgment" or child.get("intent_id") != CHILD
            or child.get("intent_name") != CHILD or child.get("class") != "judgment" or child.get("evaluation", {}).get("kind") != "judgment"):
        raise ValueError("fresh family/child agreement failed")
    for key in ("request", "response", "onchain"):
        if isinstance(child.get(key), dict) and child[key].get("served_by") not in (None, "", FAMILY):
            raise ValueError("advertised schema belongs to another family")
    return "judgment"


def capture(root):
    root.mkdir(parents=True, exist_ok=True)
    snapshot = {"schema": 1}
    for key, url in URLS.items():
        request = urllib.request.Request(url, headers={"User-Agent": "academic-source-class/1", "Cache-Control": "no-cache"})
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
            raw = response.read((1 << 20) + 1)
            if response.status != 200 or len(raw) > 1 << 20 or "json" not in response.headers.get("Content-Type", ""):
                raise ValueError("invalid authoritative class response")
        snapshot[key] = {"url": url, "raw": raw.decode()}
    snapshot["fetched_at"] = datetime.now(timezone.utc).isoformat()
    validate(snapshot)
    encoded = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode()
    (root / "class-snapshot.json").write_bytes(encoded)
    report = {"family": FAMILY, "child": CHILD, "class": "judgment", "fetched_at": snapshot["fetched_at"],
              "snapshot_sha256": sha(encoded), "pool_rows_read": False}
    (root / "class-capture.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(capture(parser.parse_args().root), indent=2))
