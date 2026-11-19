#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
HERE="$(dirname "$(realpath "$0")")"
OUT="${1:-$HERE/.verify-output}"
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1
python3 -W error -m unittest discover -s "$HERE" -p 'test_*.py' -v
node --check "$HERE/execute.mjs"
if ! test -d "$HERE/node_modules"; then
  npm --prefix "$HERE" ci --ignore-scripts --no-fund
fi
npm --prefix "$HERE" audit --audit-level=low
python3 -W error "$HERE/data.py" --out "$OUT/data"
python3 -W error "$HERE/contracts.py" --inventory "$OUT/data/inventory.json"
timeout --signal=TERM --kill-after=5s 180 node "$HERE/execute.mjs" "$OUT/data/inventory.json" "$OUT/evm-report.json"
python3 -W error "$HERE/shortcuts.py" --inventory "$OUT/data/inventory.json" --frozen "$HERE/frozen-baselines.json" --out "$OUT/shortcut-report.json"
python3 -W error "$HERE/failure_controls.py" --root "$OUT"
python3 -W error "$HERE/license_gate.py" --root "$OUT"
printf '%s\n' 'ALL GREEN, non-neural source/build gate only'
