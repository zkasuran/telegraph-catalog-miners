# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Explicit owner recipe freeze, no fitting or model operation."""
from build import HERE, sha, write_json


def main():
    paths = [p for p in HERE.rglob("*") if p.is_file() and p.name != "freeze.json" and p.name != "live-preflight.json"]
    write_json(HERE / "freeze.json", {"SPDX-License-Identifier": "LicenseRef-zkasuran-SAND-1.0", "schema_version": 1, "family": "WEATHER_FORECAST_VERIFY",
               "run_id": "cb65961105934a849aceec6d66add0db", "audit_metrics_opened": False,
               "files": {str(p.relative_to(HERE)): sha(p.read_bytes()) for p in sorted(paths)}})
    print("Frozen source/data/recipe files:", len(paths))


if __name__ == "__main__":
    main()
