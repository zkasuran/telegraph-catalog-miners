#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "$0")"
# The same source gate runs locally and in the public workflow. Tests explicitly
# isolate laptop-only negative controls from the runner's ambient context.
PYTHONDONTWRITEBYTECODE=1 python3 -W error -c 'from pathlib import Path; import sys; root=Path(sys.argv[1]); required=("LICENSE","NOTICE","DATA-SOURCES.md","SECURITY.md","ready_contract.py","test_ready_contract.py"); missing=[p for p in required if not (root/p).is_file()]; assert not missing, missing; print("Standalone rights, threat-model and ready-contract files PASS")' "$ROOT"
PYTHONDONTWRITEBYTECODE=1 python3 -W error -c 'from pathlib import Path; import sys; paths=list(Path(sys.argv[1]).glob("*.py")); [compile(p.read_text(), str(p), "exec") for p in paths]; print("Python compilation PASS:", len(paths), "files")' "$ROOT"
PYTHONDONTWRITEBYTECODE=1 python3 -W error -m unittest discover -s "$ROOT" -p 'test_*.py' -v
printf '%s\n' 'LOCAL SOURCE AND LEXICAL CHECKS PASS. Neural candidate and node qualification remain unverified.'
