# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded public GETs only. No credentials and no model operations."""
import argparse
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FAMILY = "EVENT_OUTCOME_RESOLUTION"
CHILD = FAMILY + "_GENERAL"
KIT = "1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd"
TERMS = "https://www.nobelprize.org/about/terms-of-use-for-api-nobelprize-org-and-data-nobelprize-org/"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def fetch(url, path, receipts, ceiling=4 << 20):
    if not url.startswith(("https://api.nobelprize.org/", "https://www.nobelprize.org/",
                           "https://devnode.telegraphprotocol.com/engine/v1/intents/",
                           "https://raw.githubusercontent.com/telegraphprotocol/telegraph-wasm-baseline/")):
        raise ValueError("unapproved public source")
    req = urllib.request.Request(url, headers={"User-Agent": "HistoricalSettlementResearch/1.0"})
    with urllib.request.urlopen(req, timeout=45) as response:
        raw = response.read(ceiling + 1)
        if len(raw) > ceiling or response.status != 200:
            raise ValueError("source size/status ceiling")
        receipts.append({"url": url, "final_url": response.url, "status": response.status,
                         "bytes": len(raw), "sha256": sha(raw), "path": path.name})
    path.write_bytes(raw)
    return raw


def collect(out):
    out.mkdir(parents=True, exist_ok=True)
    receipts = []
    raw = fetch("https://devnode.telegraphprotocol.com/engine/v1/intents/" + CHILD,
                out / "live-child.json", receipts)
    child = json.loads(raw)
    if child.get("class") != "judgment" or child.get("intent_id") != CHILD:
        raise ValueError("live child class/identity mismatch")
    write(out / "task-contract.json", {k: v for k, v in child.items() if k not in ("miners", "miner_count")})
    print(json.dumps({k: v for k, v in child.items() if k not in ("miners", "miner_count")}, indent=2))
    fetch(TERMS, out / "nobel-terms.html", receipts)
    fetch("https://www.nobelprize.org/about/developer-zone-2/", out / "nobel-api-docs.html", receipts)
    base = "https://raw.githubusercontent.com/telegraphprotocol/telegraph-wasm-baseline/" + KIT + "/"
    for name in ("LICENSE", "float_runtime/train_adapter.py", "pkg/engine/llm/client.go",
                 "cmd/adapterexec/main.go", "pkg/wasm/remote/remote.go",
                 "pkg/scoring/responseparser/onchain.go", "pkg/scoring/responseparser/prompt.go"):
        fetch(base + name, out / ("kit-" + name.replace("/", "_")), receipts)
    records, total, offset = [], None, 0
    for page in range(20):
        obj = json.loads(fetch("https://api.nobelprize.org/2.1/nobelPrizes?limit=100&offset=" + str(offset),
                               out / f"prizes-{page:02}.json", receipts))
        meta, rows = obj["meta"], obj["nobelPrizes"]
        if meta["offset"] != offset or not isinstance(rows, list) or not rows:
            raise ValueError("invalid pagination or empty incomplete page")
        if total is None:
            total = meta["count"]
        if meta["count"] != total or total > 2000:
            raise ValueError("source inventory changed or exceeded bound")
        records.extend(rows)
        offset += len(rows)
        if offset == total:
            break
        if offset > total:
            raise ValueError("inventory count overrun")
        time.sleep(0.2)
    else:
        raise ValueError("pagination ceiling")
    ids = {(r["awardYear"], r["category"]["en"]) for r in records}
    if len(ids) != total:
        raise ValueError("duplicate/missing source events")
    write(out / "prizes.json", records)
    write(out / "receipts.json", {"fetched_at": datetime.now(timezone.utc).isoformat(), "sources": receipts,
                                  "inventory_records": len(records), "reported_count": total,
                                  "unique_year_categories": len(ids), "kit_commit": KIT})
    print(f"Complete inventory: {len(records)} prize records, {len(ids)} distinct year/categories")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    collect(parser.parse_args().out)
