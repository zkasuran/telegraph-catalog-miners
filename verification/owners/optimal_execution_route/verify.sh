#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(realpath "$(dirname "$0")")"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$ROOT/.gate-work"
python3 -W error "$ROOT/source_import_guard.py"
export PYTHONPATH="$ROOT/.gate-work/source-guard:$ROOT:$ROOT/.gate-work/tools${PYTHONPATH:+:$PYTHONPATH}"
if ! python3 -c 'import pyflakes, yaml; assert pyflakes.__version__ == "3.4.0"; assert yaml.__version__ == "6.0.3"' >/dev/null 2>&1; then
  python3 -m pip --isolated install --no-input --no-deps --require-hashes --only-binary=:all: --index-url https://pypi.org/simple --target "$ROOT/.gate-work/tools" -r "$ROOT/requirements-source.lock"
fi
python3 -W error "$ROOT/release_gate.py"
printf '%s\n' 'ALL GREEN: source contracts and literal-wire information controls. Qualification remains gated.'
