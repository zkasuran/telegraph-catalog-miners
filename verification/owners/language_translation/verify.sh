#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "$(realpath "$0")")"
export PYTHONDONTWRITEBYTECODE=1
python3 -W error -m unittest discover -s "$ROOT" -p 'test_*.py' -v
python3 -W error "$ROOT/verify_source.py"
if command -v ruff >/dev/null 2>&1; then
  ruff check --no-cache "$ROOT"
fi
printf '%s\n' 'SOURCE RECOVERY GATE PASS. NO MODEL EXECUTION OR CANONICAL CLAIM.'
