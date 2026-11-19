#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "$(realpath "$0")")"
export PYTHONDONTWRITEBYTECODE=1
python3 -W error -m unittest discover -s "$ROOT" -p 'test_diagnostic.py' -v
python3 - "$ROOT" <<'PY'
import ast
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
for path in root.glob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
json.loads((root / 'contract.json').read_text())
print('Diagnostic source syntax and contract JSON passed')
PY
