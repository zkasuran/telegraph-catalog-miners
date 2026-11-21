# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Exercise the actual bound source inputs, not a model or a registration report."""
import json
from support import load_inputs


def main():
    spec, rows, report = load_inputs()
    print(json.dumps({"source_pairs": {name: len(values) for name, values in rows.items()},
                      "development_only": spec["development_only"], "independent_audit": report["independent_audit"],
                      "neural_operations": False}, sort_keys=True))


if __name__ == "__main__":
    main()
