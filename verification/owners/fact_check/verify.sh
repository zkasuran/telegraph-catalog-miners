#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
OWNER_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
python3 -W error -m unittest discover -s "$OWNER_DIR" -p 'test_*.py' -v
python3 -W error "$OWNER_DIR/release_gate.py"
python3 -W error "$OWNER_DIR/development_source_gate.py"
python3 -W error "$OWNER_DIR/inventory_replay.py"
python3 -W error -c "import ast,pathlib; root=pathlib.Path('$OWNER_DIR'); [compile(ast.parse((root/name).read_text()),name,'exec') for name in ('recover.py','recovery_policy.py','converge.py','convergence_policy.py')]"
printf 'ALL GREEN\n'
