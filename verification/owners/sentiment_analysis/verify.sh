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
v3 = json.loads((root / 'v3-config.json').read_text())
for key in ('source_manifest_sha256', 'sources_sha256', 'baseline_sha256', 'quarantine_sha256',
            'train_sha256', 'validation_sha256', 'audit_sha256', 'warm_package_sha256'):
    if not isinstance(v3.get(key), str) or not re.fullmatch('[a-f0-9]{64}', v3[key]):
        raise ValueError('missing frozen v3 source/checkpoint identity: ' + key)
if v3['source_version'] != 3 or not (1 <= v3['hyperparameters']['epochs'] <= 3 and
                                   v3['hyperparameters']['gate_min_acc'] >= .8):
    raise ValueError('invalid bounded v3 configuration')
for name in ('SENTIMENT_ANALYSIS-v3.safetensors', 'v3-registration-ready.json',
             'v3-source-manifest.json', 'v3-asset-notice.json'):
    if not (root / 'model' / name).is_file():
        raise ValueError('passing immutable release asset missing: ' + name)
recovery = json.loads((root / 'endpoint-recovery-plan.json').read_text())
if (recovery['origin_package_sha256'] != 'ae2d3b0275e4f96419bee0e75396f445003afbfc401029353e5bc8ef9e3a061e' or
        recovery['bounds']['lease_seconds'] != 18000 or recovery['bounds']['remote_job_minutes'] > 350 or
        recovery['bounds']['guard_seconds'] > 86400 or recovery['bounds']['max_generations'] > 6):
    raise ValueError('recovery/continuity must retain reviewed source and bounded lease policy')
for name in ('LICENSE', 'NOTICE', 'DATA-SOURCES.md', 'SECURITY.md'):
    if not (root / name).is_file() or not (root / name).read_text().strip():
        raise ValueError('required standalone rights/threat-model artifact missing: ' + name)
print('Standalone source syntax, JSON and rights checks passed')
PY
if command -v ruff >/dev/null 2>&1; then
    ruff check --no-cache "$ROOT"
fi
printf '%s\n' 'SENTIMENT OWNER SOURCE GREEN'
