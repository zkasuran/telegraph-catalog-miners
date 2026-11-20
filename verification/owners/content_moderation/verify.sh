#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "$(realpath "$0")")"
export PYTHONDONTWRITEBYTECODE=1
bash -n "$ROOT/verify.sh"
python3 -W error "$ROOT/verify_source.py"
python3 -W error "$ROOT/check_checkpoint_terms.py"
python3 -W error -m unittest discover -s "$ROOT" -p 'test_owner.py' -v
python3 -W error -m unittest discover -s "$ROOT" -p 'test_recovery.py' -v
python3 -W error -m unittest discover -s "$ROOT" -p 'test_balanced.py' -v
