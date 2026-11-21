#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
HERE="$(dirname "$(realpath "$0")")"
OUT="${1:-$HERE/.verify-output}"
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1
python3 -W error -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); compile(p.read_text(),str(p),"exec"); print("Parser diagnostic source compiled without neural imports")' "$HERE/diagnose_parser_remote.py"
python3 -W error "$HERE/policy.py" --check-public-contract
python3 -W error -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); compile(p.read_text(),str(p),"exec"); print("Live class source adapter syntax compiled without model imports")' "$HERE/class_source.py"
python3 -W error -m unittest discover -s "$HERE" -p 'test_*.py' -v
PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" python3 -W error -c 'from objective import load; import sys; c,h=load(sys.argv[1]); assert (c["ranking_numerator"],c["calibration_numerator"],c["denominator"])==(1,1,2); print("Frozen joint objective validated without model imports: "+h)' "$HERE/objective-v1.json"
python3 -W error -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); compile(p.read_text(),str(p),"exec"); print("Exact validation gate-vector selector compiled without neural imports")' "$HERE/selection.py"
PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" python3 -W error -c 'from regularization import load; import sys; d,h=load(sys.argv[1]); assert d==0.25; print("Frozen adapter dropout validated without neural imports: "+str(d)+" "+h)' "$HERE/regularization-v1.json"
python3 -W error "$HERE/trainer_selection.py" --source-check-report "$OUT/trainer-selection-contract.json"
node --check "$HERE/execute.mjs"
if ! test -d "$HERE/node_modules"; then
  npm --prefix "$HERE" ci --ignore-scripts --no-fund
fi
npm --prefix "$HERE" audit --audit-level=low
python3 -W error "$HERE/data.py" --out "$OUT/data"
PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" python3 -W error -c 'from data import response_views; from pathlib import Path; import json,sys; r=json.loads(Path(sys.argv[1]).read_text()); t=response_views([x for x in r if x["split"]=="train"],"train"); v=response_views([x for x in r if x["split"]=="validation"],"validation"); assert len(t)==504 and len(v)==72; assert not any(x["template_group"].startswith("design-") and x["split"]=="audit" for x in r); print("Broad development: 504 train views, unchanged validation and clean reserved audit; exposed audit excluded")' "$OUT/data/inventory.json"
python3 -W error "$HERE/contracts.py" --inventory "$OUT/data/inventory.json"
python3 -W error "$HERE/sampling.py" --train "$OUT/data/train.jsonl" --frozen "$HERE/frozen-sampling-v5.json" --report "$OUT/training-sampling-contract.json"
python3 -W error "$HERE/paired_sampling.py" --train "$OUT/data/train.jsonl" --frozen "$HERE/frozen-sampling-v5.json" --report "$OUT/paired-batch-contract.json"
PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" python3 -W error -c 'from contracts import heldout_pairs; rows=[{"split":"validation"},{"split":"audit"}]; p,allowed=heldout_pairs(rows,"validation"); assert len(p)==1 and not allowed; print("Validation source boundary excludes audit and registration-ready claims")'
timeout --signal=TERM --kill-after=5s 180 node "$HERE/execute.mjs" "$OUT/data/inventory.json" "$OUT/evm-report.json"
python3 -W error "$HERE/shortcuts.py" --inventory "$OUT/data/inventory.json" --frozen "$HERE/frozen-baselines-v6.json" --out "$OUT/shortcut-report.json"
python3 -W error "$HERE/failure_controls.py" --root "$OUT"
python3 -W error "$HERE/license_gate.py" --root "$OUT"
printf '%s\n' 'ALL GREEN, non-neural source/build gate only'
