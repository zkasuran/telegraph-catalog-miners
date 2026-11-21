#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
here="$(dirname "$(realpath "$0")")"
export PYTHONDONTWRITEBYTECODE=1
parent="${OWNER_VERIFY_TEMP:-/tmp/opencode}"
mkdir -p "$parent"
work="$(mktemp -d "$parent/email-security-verify.XXXXXX")"
trap 'rm -rf "$work"' EXIT
python="${OWNER_VERIFY_PYTHON:-python3}"
if ! "$python" -c 'import ruff, yaml' >/dev/null 2>&1; then
  python3 -m venv "$work/tools"
  python="$work/tools/bin/python"
  PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple "$python" -m pip install --disable-pip-version-check --require-hashes -r "$here/requirements-verify.lock"
fi
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
"$python" -W error "$here/data.py" --out "$work/data"
"$python" -W error "$here/static_check.py" --data "$work/data"
"$python" -W error -m unittest discover -s "$here" -p 'test_*.py' -v
"$python" -m ruff check --no-cache --select E,F --ignore E501 "$here"
"$python" -W error "$here/release_gate.py"
printf 'ALL GREEN source and non-neural consistency gates only\n'
