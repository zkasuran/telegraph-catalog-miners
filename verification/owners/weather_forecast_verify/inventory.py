# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Read complete station inventory and complete selected station records."""
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
from capture import capture

HERE = Path(__file__).resolve().parent
BASE = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/"
# Named before retrieval. This is a bounded cross-region subset, not the corpus.
STATIONS = ("USW00094728", "USW00012839", "USW00094846", "USW00024233",
            "USW00023050", "USW00013876")


def main():
    root = HERE / "sources"
    manifest = json.loads((root / "capture-manifest.json").read_text())
    files = ("ghcnd-stations.txt",) + tuple("all/" + s + ".dly" for s in STATIONS)
    report = {"scope": "six preselected US station files, complete records; not all GHCN", "stations": {}}
    for name in files:
        target = root / name.replace("/", "-")
        if target.exists():
            raw = target.read_bytes()
        else:
            raw, _ = capture(BASE + name, 32 << 20)
        if target.name in manifest and hashlib.sha256(raw).hexdigest() != manifest[target.name]["sha256"]:
            raise ValueError("frozen station source changed: " + name)
        if not target.exists():
            target.write_bytes(raw)
        if target.name not in manifest:
            manifest[target.name] = {"url": BASE + name, "bytes": len(raw),
                                     "sha256": hashlib.sha256(raw).hexdigest(),
                                     "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        (root / "capture-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if name == "ghcnd-stations.txt":
            lines = raw.decode("ascii").splitlines()
            report["full_inventory_stations"] = len(lines)
            report["selected_station_metadata"] = [x for x in lines if x[:11] in STATIONS]
            if len(report["selected_station_metadata"]) != len(STATIONS):
                raise ValueError("selected station absent or duplicated")
        else:
            counts, years = Counter(), set()
            lines = raw.decode("ascii").splitlines()
            for line in lines:
                if len(line) != 269 or line[:11] != name[4:15]:
                    raise ValueError("malformed or wrong station record")
                if line[17:21] not in ("TMAX", "TMIN", "PRCP"):
                    continue
                for offset in range(21, 269, 8):
                    value = int(line[offset:offset + 5])
                    if value != -9999:
                        counts[line[offset + 7]] += 1
                        if line[offset + 7] == "1":
                            years.add(int(line[11:15]))
            report["stations"][name[4:15]] = {"month_records": len(lines), "source_counts": dict(counts),
                                              "cf6_years": sorted(years)}
        (root / "inventory-report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(target.name, len(raw), "bytes")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
