#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "$(realpath "$0")")"
export PYTHONDONTWRITEBYTECODE=1
python3 -W error -m unittest discover -s "$ROOT" -p 'test_*.py' -v
ruff check --no-cache "$ROOT"
ruff format --check "$ROOT"
python3 -W error - "$ROOT" <<'PY'
import sys
from pathlib import Path
for path in Path(sys.argv[1]).glob('*.py'):
    compile(path.read_bytes(), str(path), 'exec')
print('General preference source contracts and compilation pass. No neural work.')
PY
