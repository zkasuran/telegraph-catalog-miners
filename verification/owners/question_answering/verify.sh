#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
HERE="$(dirname "$(realpath "$0")")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
python3 -W error -m unittest discover -s "$HERE" -p 'test_contracts.py' -v
python3 -W error "$HERE/static_check.py"
python3 -c 'import sys; assert not {"torch","transformers","peft","onnxruntime","tensorflow"}.intersection(sys.modules); print("ALL GREEN, source-only gate")'
