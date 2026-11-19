# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Reverify only the public live task. Never read evaluation-pool rows."""
import json
from datetime import datetime, timezone
import urllib.request
from build import HERE, sha, write_json
from capture import capture


def capture_classes(root):
    envelope = {"schema": 1, "fetched_at": datetime.now(timezone.utc).isoformat()}
    for name, url in (("family", "https://devnode.telegraphprotocol.com/api/wasm?resolve=WEATHER_FORECAST_VERIFY"),
                      ("child", "https://devnode.telegraphprotocol.com/engine/v1/intents/WEATHER_FORECAST_VERIFY_GENERAL")):
        request = urllib.request.Request(url, headers={"User-Agent": "Weather-owner-class/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read((2 << 20) + 1)
        if len(raw) > 2 << 20:
            raise ValueError("class response ceiling")
        envelope[name] = {"url": url, "raw": raw.decode()}
    from lease_common import check_classes
    check_classes(envelope, {"family": "WEATHER_FORECAST_VERIFY", "child": "WEATHER_FORECAST_VERIFY_GENERAL", "class": "judgment"})
    write_json(root / "class-snapshot.json", envelope)
    return envelope


def main():
    raw, final = capture("https://devnode.telegraphprotocol.com/engine/v1/intents/WEATHER_FORECAST_VERIFY_GENERAL", 1 << 20)
    spec = json.loads(raw)
    original = json.loads((HERE / "sources/intent-projection.json").read_text())
    if spec.get("class") != "judgment" or spec.get("intent_id") != "WEATHER_FORECAST_VERIFY_GENERAL":
        raise ValueError("live family or child class changed")
    # Literal published schemas, not a normalized stand-in. Miner order is not a schema.
    if spec.get("request") != original["request"] or spec.get("response") != original["response"]:
        raise ValueError("literal live weather task schema changed")
    if spec.get("lifecycle", {}).get("state") != "active":
        raise ValueError("live child not active")
    report = {"family": "WEATHER_FORECAST_VERIFY", "class": spec["class"], "url": final,
              "sha256": sha(raw), "bytes": len(raw), "schema_matches_frozen_task": True,
              "public_definition_only": True, "evaluation": spec["evaluation"]}
    write_json(HERE / "live-preflight.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
