#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "$(realpath "$0")")"
export PYTHONDONTWRITEBYTECODE=1
python3 -W error -m unittest discover -s "$ROOT" -p 'test_*.py' -v
python3 - "$ROOT" <<'PY'
import ast
import json
import re
import sys
from pathlib import Path
root = Path(sys.argv[1])
for path in root.glob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
for path in root.glob('*.json'):
    json.loads(path.read_text())
config = json.loads((root / 'retrain-config.json').read_text())
for key in ('source_manifest_sha256', 'sources_sha256', 'baseline_sha256',
            'train_sha256', 'validation_sha256', 'audit_sha256', 'warm_package_sha256'):
    if not isinstance(config.get(key), str) or not re.fullmatch('[a-f0-9]{64}', config[key]):
        raise ValueError('missing frozen retraining hash: ' + key)
hp = config['hyperparameters']
if not (1 <= hp['max_length'] <= 512 and 1 <= hp['epochs'] <= 3 and
        1 <= hp['lora_rank'] <= 16 and 1 <= hp['batch_size'] <= 4 and hp['gate_min_acc'] >= .8):
    raise ValueError('retraining config exceeds bounded quality/runtime policy')
for name in ('LICENSE', 'NOTICE', 'DATA-SOURCES.md', 'SECURITY.md'):
    if not (root / name).is_file() or not (root / name).read_text().strip():
        raise ValueError('required standalone rights/threat-model artifact missing: ' + name)
print('Standalone source syntax, JSON and rights checks passed')
PY
if command -v ruff >/dev/null 2>&1; then
    ruff check --no-cache "$ROOT"
fi
printf '%s\n' 'SENTIMENT OWNER SOURCE GREEN'
