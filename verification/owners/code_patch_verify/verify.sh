#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
HERE="$(dirname "$(realpath "$0")")"
export PYTHONDONTWRITEBYTECODE=1
python3 -W error "$HERE/verify.py"
python3 -W error "$HERE/ci_reference.py" "$HERE/data/smoke.json"
if python3 -W error "$HERE/ci_reference.py" "$HERE/data/tampered.json"; then
    exit 1
fi
python3 -W error "$HERE/remote.py" --check-only
