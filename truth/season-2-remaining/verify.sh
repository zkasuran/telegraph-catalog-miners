#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "$(realpath "$0")")"
export PYTHONDONTWRITEBYTECODE=1
python3 -W error "$ROOT/check-package.py"
node --check "$ROOT/worker.mjs"
node --check "$ROOT/sources/finance/finance.mjs"
node --check "$ROOT/sources/operations/operations.mjs"
node --test "$ROOT/test-integration.mjs"
python3 -W error "$ROOT/baseline-gate.py"
python3 -W error "$ROOT/check-package.py"
printf '%s\n' 'ALL GREEN: portable integration release checks. Native loader and admission are unverified.'
