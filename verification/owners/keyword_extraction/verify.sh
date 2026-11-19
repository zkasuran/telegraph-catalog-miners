#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "$0")"
PYTHONDONTWRITEBYTECODE=1 python3 -W error -c 'from pathlib import Path; import sys; paths=list(Path(sys.argv[1]).glob("*.py")); [compile(p.read_text(), str(p), "exec") for p in paths]; print("Python compilation PASS:", len(paths), "files")' "$ROOT"
PYTHONDONTWRITEBYTECODE=1 python3 -W error -m unittest discover -s "$ROOT" -p 'test_keyword.py' -v
printf '%s\n' 'LOCAL SOURCE AND LEXICAL CHECKS PASS. Neural candidate and node qualification remain unverified.'
