#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "$(realpath "$0")")"
export PYTHONDONTWRITEBYTECODE=1
python3 -W error "$ROOT/verify.py"
python3 -W error -m unittest discover -s "$ROOT" -p 'test_*.py' -v
printf '%s\n' 'ALL GREEN, source-only. No local neural operation or canonical claim.'
