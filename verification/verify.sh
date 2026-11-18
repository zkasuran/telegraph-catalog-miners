#!/usr/bin/env bash
set -euo pipefail
ROOT="$(dirname "$(realpath "$0")")"
export PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -s "$ROOT" -p 'test_*.py'
python3 -m unittest discover -s "$ROOT/training" -p 'test_*.py'
python3 - "$ROOT" <<'PY'
import ast
import sys
from pathlib import Path
for path in Path(sys.argv[1]).rglob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
import json
for path in Path(sys.argv[1]).rglob('*.json'):
    json.loads(path.read_text())
print('Python syntax checks passed')
PY
