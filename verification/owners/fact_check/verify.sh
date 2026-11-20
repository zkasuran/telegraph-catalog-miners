#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
OWNER_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
python3 -W error -m unittest discover -s "$OWNER_DIR" -p 'test_*.py' -v
python3 -W error "$OWNER_DIR/release_gate.py"
python3 -W error "$OWNER_DIR/development_source_gate.py"
printf 'ALL GREEN\n'
