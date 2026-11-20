#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
HERE="$(dirname "$(realpath "$0")")"
export PYTHONDONTWRITEBYTECODE=1
export GOMAXPROCS=2
GATE="$(mktemp -d "${TMPDIR:-/tmp}/summary-source-gate.XXXXXX")"
trap 'rm -rf "$GATE"' EXIT
bash -n "$HERE/verify.sh"
if [[ -n "${RUFF:-}" ]]; then
  RUFF_BIN="$RUFF"
elif command -v ruff >/dev/null 2>&1 && [[ "$(ruff --version)" == "ruff 0.16.4" ]]; then
  RUFF_BIN="$(command -v ruff)"
else
  python3 -m venv "$GATE/venv"
  timeout 120 "$GATE/venv/bin/python" -m pip install --require-hashes --only-binary=:all: --index-url https://pypi.org/simple -r "$HERE/requirements-source.lock"
  RUFF_BIN="$GATE/venv/bin/ruff"
fi
[[ "$("$RUFF_BIN" --version)" == "ruff 0.16.4" ]]
"$RUFF_BIN" check --config "$HERE/ruff.toml" "$HERE"
python3 -W error "$HERE/static_check.py"
python3 -W error "$HERE/audit_policy.py"
python3 -W error "$HERE/diagnostic_source_check.py"
python3 -W error "$HERE/curriculum.py"
python3 -W error "$HERE/reciprocal_source_check.py"
python3 -W error "$HERE/role_bootstrap.py"
python3 -W error "$HERE/frozen_candidate.py"
python3 -W error "$HERE/released_report.py"
python3 -W error -m unittest discover -s "$HERE" -p 'test_*.py' -v
KIT="${REFERENCE_KIT:-}"
if [[ -z "$KIT" ]]; then
  KIT="$(python3 -W error "$HERE/reference.py" --kit-only --root "$GATE")"
fi
[[ -z "$(gofmt -l "$HERE/packagecheck.go" "$HERE/packagecheck_test.go" "$HERE/wirecheck_test.go")" ]]
timeout 600 go -C "$KIT" test -p=1 -mod=readonly -v "$HERE/packagecheck.go" "$HERE/packagecheck_test.go" "$HERE/wirecheck_test.go"
printf '%s\n' 'ALL GREEN, source-only. No model or node result claimed.'
