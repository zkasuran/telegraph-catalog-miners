#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
python3 -W error -m unittest discover -s "$ROOT" -p 'test_wirecheck.py' -v
python3 -W error "$ROOT/wirecheck.py" --help >/dev/null
printf '%s\n' 'ALL GREEN: constructed report controls only, no model or reference engine'
