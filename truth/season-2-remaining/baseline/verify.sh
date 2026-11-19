#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "$(realpath "$0")")"
PYTHONDONTWRITEBYTECODE=1 python3 -W error "$ROOT/check-package.py"
node --check "$ROOT/worker.mjs"
node --check "$ROOT/dispatcher.mjs"
node --check "$ROOT/workers-fetch.mjs"
mkdir -p "$ROOT/.test-results"
node --test "$ROOT/test-dispatcher.mjs"
node --test "$ROOT/sources/corporate/test_worker.mjs"
node --test "$ROOT/sources/onchain/test-worker.mjs"
node --test "$ROOT/sources/hashprice/test-worker.mjs" "$ROOT/sources/hashprice/test-retry.mjs" "$ROOT/sources/hashprice/test-envelope.mjs"
PYTHONDONTWRITEBYTECODE=1 python3 -W error "$ROOT/sources/hashprice/envelope-mutations.py"
node "$ROOT/mutation-check.mjs"
PYTHONDONTWRITEBYTECODE=1 python3 -W error "$ROOT/check-package.py"
printf '%s\n' 'PORTABLE REPLAY GATE PASS. No deployment or actual node admission claimed.'
