#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "$(realpath "$0")")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/vendor${PYTHONPATH:+:$PYTHONPATH}"
python3 -W error "$ROOT/gate.py"
python3 -W error -m pyflakes "$ROOT/scorer.py" "$ROOT/inventory.py" "$ROOT/split_gate.py" "$ROOT/test_product.py" "$ROOT/source_inventory.py" "$ROOT/product_evidence.py" "$ROOT/test_expansion.py" "$ROOT/gate.py"
python3 -W error -m unittest discover -s "$ROOT" -p 'test_*.py' -v
printf '%s\n' 'ALL GREEN: archived source-report diagnostic. Model qualification is not ready.'
