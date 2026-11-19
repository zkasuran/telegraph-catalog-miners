# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Capture actual live model classes only. No evaluation-pool data or credentials."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

FAMILY = "SMART_CONTRACT_AUDIT"
CHILD = FAMILY + "_GENERAL"


def validate(envelope):
    if envelope.get("schema") != 1 or envelope["family"]["url"] != "https://devnode.telegraphprotocol.com/api/wasm?resolve=" + FAMILY or envelope["child"]["url"] != "https://devnode.telegraphprotocol.com/engine/v1/intents/" + CHILD:
        raise ValueError("exact live class sources required")
    fetched = datetime.fromisoformat(envelope["fetched_at"].replace("Z", "+00:00"))
    age = (datetime.now(timezone.utc) - fetched).total_seconds()
    if not -60 <= age <= 900:
        raise ValueError("fresh live class sources required")
    parent = json.loads(envelope["family"]["raw"])["resolve"]
    child = json.loads(envelope["child"]["raw"])
    if parent["intent"] != FAMILY or parent["class"] != "judgment" or child["intent_id"] != CHILD or child["intent_name"] != CHILD or child["class"] != parent["class"] or child["evaluation"]["kind"] != parent["class"]:
        raise ValueError("actual live family/child/model class mismatch")
    return parent, child


def capture(path):
    envelope = {"schema": 1, "fetched_at": datetime.now(timezone.utc).isoformat()}
    for key, url in (("family", "https://devnode.telegraphprotocol.com/api/wasm?resolve=" + FAMILY),
                     ("child", "https://devnode.telegraphprotocol.com/engine/v1/intents/" + CHILD)):
        with urlopen(Request(url, headers={"User-Agent": "source-bound-class-check/1", "Cache-Control": "no-cache"}), timeout=30) as response:
            raw = response.read((2 << 20) + 1)
        if len(raw) > 2 << 20:
            raise ValueError("class source byte ceiling")
        envelope[key] = {"url": url, "raw": raw.decode()}
    parent, child = validate(envelope)
    raw = (json.dumps(envelope, indent=2) + "\n").encode()
    path.write_bytes(raw)
    result = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "family": FAMILY, "child": CHILD,
              "class": parent["class"], "child_miner_count": child.get("miner_count"), "typed_schema_ready": child.get("onchain", {}).get("ready"),
              "source_scope": "Live classes only. Null miner schema is not a complete scoring rubric or automatic model-class refusal."}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    result = capture(parser.parse_args().out)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
