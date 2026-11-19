#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
python3 -W error "$(dirname "$0")/release_gate.py"
