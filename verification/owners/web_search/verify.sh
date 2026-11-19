#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
HERE="$(dirname "$(realpath "$0")")"
export PYTHONDONTWRITEBYTECODE=1
python3 -W error "$HERE/verify_source.py"
python3 -W error -m unittest discover -s "$HERE" -p 'test_*.py' -v
printf '%s\n' 'SOURCE GATE PASSED. NO MODEL RESULT CLAIMED.'
