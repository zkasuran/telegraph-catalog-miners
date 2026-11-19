#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "$(realpath "$0")")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/vendor${PYTHONPATH:+:$PYTHONPATH}"
python3 -W error "$ROOT/gate.py"
python3 -W error -m pyflakes "$ROOT/scorer.py" "$ROOT/inventory.py" "$ROOT/split_gate.py" "$ROOT/test_product.py" "$ROOT/source_inventory.py" "$ROOT/product_evidence.py" "$ROOT/test_expansion.py" "$ROOT/gate.py"
python3 -W error -m unittest discover -s "$ROOT" -p 'test_*.py' -v
python3 -W error -m pyflakes "$ROOT/development/data_contract.py" "$ROOT/development/source_admission.py" "$ROOT/development/source_inventory.py" "$ROOT/development/remote_develop.py" "$ROOT/development/runtime_contract.py" "$ROOT/development/test_development.py"
python3 -W error -m unittest discover -s "$ROOT/development" -p 'test_development.py' -v
bash "$ROOT/audit_admission/verify.sh"
printf '%s\n' 'ALL GREEN: archived source-report diagnostic. Model qualification is not ready.'
