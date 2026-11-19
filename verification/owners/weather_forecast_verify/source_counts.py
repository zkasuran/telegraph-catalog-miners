# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Structural source counts only, never model or audit performance."""
from collections import Counter
import json
from build import HERE, load_archive


def main():
    observations, _ = load_archive()
    counts = Counter((o.station, o.day[:7], o.element) for o in observations.values())
    result = {"monthly_source_clean_counts": {":".join(key): count for key, count in sorted(counts.items())}}
    (HERE / "sources/source-counts.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"TMAX": {":".join(key[:2]): count for key, count in sorted(counts.items()) if key[2] == "TMAX"}}, indent=2))


if __name__ == "__main__":
    main()
