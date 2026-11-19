# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded keyless primary-source capture. No model or evaluation-pool data."""
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent
PIN = "1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd"
RAW = "https://raw.githubusercontent.com/telegraphprotocol/telegraph-wasm-baseline/" + PIN + "/"
SOURCES = {
    "intent": "https://devnode.telegraphprotocol.com/engine/v1/intents/WEATHER_FORECAST_VERIFY_GENERAL",
    "nws-disclaimer": "https://www.weather.gov/disclaimer",
    "ghcn-readme": "https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt",
    "adapter-input.go": RAW + "pkg/scoring/comparator/response_format.go",
    "onchain.go": RAW + "pkg/scoring/responseparser/onchain.go",
    "remote.go": RAW + "pkg/wasm/remote/remote.go",
    "train_adapter.py": RAW + "float_runtime/train_adapter.py",
    "kit-license": RAW + "LICENSE",
}


def capture(url, ceiling=16 << 20):
    if not url.startswith(("https://devnode.telegraphprotocol.com/engine/v1/intents/", RAW,
                           "https://www.weather.gov/", "https://www.ncei.noaa.gov/pub/data/ghcn/daily/")):
        raise ValueError("source outside allowlist")
    request = urllib.request.Request(url, headers={"User-Agent": "Weather-source-verification/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(ceiling + 1)
        if len(raw) > ceiling:
            raise ValueError("source exceeds byte ceiling")
        return raw, response.geturl()


def main():
    root = HERE / "sources"
    root.mkdir(exist_ok=True)
    manifest = {}
    for name, url in SOURCES.items():
        target = root / name
        old = root / "capture-manifest.json"
        prior = json.loads(old.read_text()) if old.exists() else {}
        if target.exists():
            raw, final = target.read_bytes(), url
            print(name + ": preserved")
        else:
            raw, final = capture(url)
        if name in prior and hashlib.sha256(raw).hexdigest() != prior[name]["sha256"]:
            raise ValueError("frozen source changed: " + name)
        if not target.exists():
            target.write_bytes(raw)
        manifest[name] = {"url": url, "final_url": final, "bytes": len(raw),
                          "sha256": hashlib.sha256(raw).hexdigest(),
                          "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        print(name + ": " + str(len(raw)) + " bytes")
        old.write_text(json.dumps({**manifest, **prior}, indent=2) + "\n")
        if name == "intent":
            spec = json.loads(raw)
            projection = {k: v for k, v in spec.items() if k != "miners"}
            (root / "intent-projection.json").write_text(json.dumps(projection, indent=2) + "\n")


if __name__ == "__main__":
    main()
