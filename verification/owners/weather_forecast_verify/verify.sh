#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
LINT_TEMP=""
if ! command -v "${RUFF:-ruff}" >/dev/null 2>&1; then
    LINT_TEMP="$(mktemp -d)"
    trap 'if [ -n "$LINT_TEMP" ]; then rm -rf "$LINT_TEMP"; fi' EXIT
    timeout 90 python3 -m pip install --target "$LINT_TEMP" --require-hashes --only-binary=:all: --no-deps --index-url https://pypi.org/simple -r "$ROOT/lint-requirements.txt"
    export RUFF="$LINT_TEMP/bin/ruff"
fi
"${RUFF:-ruff}" check "$ROOT" --exclude sources --select E4,E7,E9,F
python3 -W error "$ROOT/static_check.py"
python3 -W error -m unittest discover -s "$ROOT" -p 'test_weather.py' -v
python3 -W error "$ROOT/score.py" --location 'MIAMI INTL AP' --date '2025-09-01' --answer "$(python3 -W error "$ROOT/smoke_answer.py")"
if python3 -W error "$ROOT/score.py" --location 'MIAMI INTL AP' --date '2025-09-01' --answer '0'; then
    exit 1
fi
if python3 -W error "$ROOT/score.py" --location 'Miami' --date '2025-09-01' --answer '0'; then
    exit 1
fi
if env -u GITHUB_ACTIONS -u GITHUB_REPOSITORY -u GITHUB_WORKFLOW python3 -W error "$ROOT/remote.py" --root "$ROOT/not-a-remote-run"; then
    exit 1
fi
bash "$ROOT/wire/verify.sh"
printf '%s\n' 'ALL GREEN: source-bound local gate only, neural and node work not run'
