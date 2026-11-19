# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Useful non-neural archive diagnostic, with explicit unsupported abstention."""
import argparse
import json
from build import load_archive
from core import ArchiveScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--answer", required=True)
    args = parser.parse_args()
    observations, names = load_archive()
    # The fixed baseline exposes only the frozen historical archive period.
    observations = {key: value for key, value in observations.items() if value.day <= "2026-09-30"}
    result = ArchiveScorer(observations, names).score(args.location, args.date, args.answer)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "supported" and result["absolute_error_c"] in ("0", "0.0") else 1


if __name__ == "__main__":
    raise SystemExit(main())
