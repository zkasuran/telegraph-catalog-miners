#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "$(realpath "$0")")"
export PYTHONDONTWRITEBYTECODE=1
bash -n "$ROOT/verify.sh"
python3 -W error -m unittest discover -s "$ROOT" -p 'test_*.py' -v
python3 -W error "$ROOT/static_check.py"
printf '%s\n' 'ALL GREEN: source-only gate, retained inputs and symmetric objective controls; neural checks run remotely'
