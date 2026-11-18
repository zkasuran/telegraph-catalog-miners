#!/usr/bin/env bash
set -euo pipefail
ROOT="$(dirname "$(realpath "$0")")"
export PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -s "$ROOT" -p 'test_*.py'
python3 - "$ROOT" <<'PY'
import ast
import sys
from pathlib import Path
for path in Path(sys.argv[1]).glob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
print('Python syntax checks passed')
PY
