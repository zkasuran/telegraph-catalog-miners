#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
python3 -W error -m unittest discover -s "$ROOT" -p 'test_*.py' -v
python3 -W error "$ROOT/source_smoke.py"
if env -u GITHUB_ACTIONS -u GITHUB_REPOSITORY -u GITHUB_WORKFLOW python3 -W error "$ROOT/train.py" --root "$ROOT/not-a-model-run"; then
    exit 1
fi
printf '%s\n' 'ALL GREEN: development source contract only, no model or audit executed'
