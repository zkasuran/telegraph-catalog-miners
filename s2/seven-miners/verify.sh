#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "$(realpath "$0")")"
PYTHONDONTWRITEBYTECODE=1 python3 -W error "$ROOT/verify.py"
node --check "$ROOT/runtime/commerce/88/worker.mjs"
node --check "$ROOT/runtime/commerce/common.mjs"
node --check "$ROOT/stock.test.mjs"
node --test "$ROOT/stock.test.mjs"
node --check "$ROOT/chat.mjs"
node --check "$ROOT/chat-router.mjs"
node --test "$ROOT/chat.test.mjs"
PYTHONDONTWRITEBYTECODE=1 python3 -W error "$ROOT/verify.py"
printf '%s\n' 'ALL GREEN: stock and chat source release checks'
