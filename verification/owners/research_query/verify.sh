#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
here="$(dirname "${BASH_SOURCE[0]}")"
python3 -W error "$here/verify.py"
