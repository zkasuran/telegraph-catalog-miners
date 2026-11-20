#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "${BASH_SOURCE[0]}")"
PYTHONDONTWRITEBYTECODE=1 python3 -W error "$ROOT/verify_local.py" --source-generation 6
