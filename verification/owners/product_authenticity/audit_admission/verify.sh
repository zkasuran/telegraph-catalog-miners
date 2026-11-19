#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "$(realpath "$0")")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/../vendor${PYTHONPATH:+:$PYTHONPATH}"
python3 -W error -m pyflakes "$ROOT/audit_source_admission.py" "$ROOT/test_audit_source_admission.py"
python3 -W error -m unittest discover -s "$ROOT" -p 'test_audit_source_admission.py' -v
python3 -W error "$ROOT/audit_source_admission.py" --snapshot "$ROOT/sources"
printf '%s\n' 'SOURCE ADMISSION VERIFIED. INDEPENDENT MODEL AUDIT REFUSED.'
