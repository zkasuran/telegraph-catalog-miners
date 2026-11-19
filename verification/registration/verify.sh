#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "$(realpath "$0")")"
export PYTHONDONTWRITEBYTECODE=1
python3 -W error -m unittest discover -s "$ROOT" -p 'test_*.py' -v
python3 - "$ROOT" <<'PY'
import ast
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
for path in root.glob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
for path in root.glob('*.json'):
    json.loads(path.read_text())
print('Registration syntax and JSON checks passed')
PY
if command -v ruff >/dev/null 2>&1; then
    ruff check --no-cache "$ROOT"
fi
if command -v gofmt >/dev/null 2>&1; then
    test -z "$(gofmt -l "$ROOT/stagecheck.go")"
fi
printf '%s\n' 'REGISTRATION STATIC GREEN'
