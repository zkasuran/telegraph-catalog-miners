#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "$(realpath "$0")")"
export PYTHONDONTWRITEBYTECODE=1
python3 -W error -m unittest discover -s "$ROOT" -p 'test_*.py' -v
bash "$ROOT/preference/verify.sh"
ruff check --no-cache "$ROOT"
ruff format --check "$ROOT"
python3 -W error "$ROOT/check_public.py"
python3 - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
for path in root.glob('*.py'):
    compile(path.read_text(), str(path), 'exec')
for path in root.glob('*.json'):
    json.loads(path.read_text())
if 'torch' in sys.modules or 'transformers' in sys.modules:
    raise RuntimeError('source-only gate imported neural runtime')
print('Source syntax, JSON and no-neural-import checks passed')
PY
printf '%s\n' 'SOURCE PIPELINE CHECKS PASS. Model eligibility remains blocked.'
