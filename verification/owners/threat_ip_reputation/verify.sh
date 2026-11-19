#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
HERE="$(dirname -- "${BASH_SOURCE[0]}")"
PYTHONDONTWRITEBYTECODE=1 python3 -W error "$HERE/verify.py"
PYTHONDONTWRITEBYTECODE=1 python3 -W error "$HERE/learn/verify.py"
# Metadata-only recovery contracts are included in learn/verify.py.
