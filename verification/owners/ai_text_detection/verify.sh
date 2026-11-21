#!/usr/bin/env bash
# Direct diagnostic gap algebra retains the original FP32 comparison tolerances.
# Complete finite measurement never overrides an original gradient refusal.
# Format curriculum replays only existing training pairs under unchanged gates.
# Joint replay keeps original source weights and uses explicit uniform replay weights.
# Frozen-head fitting runs only remotely and retains all qualification gates.
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
ROOT="$(dirname "$(realpath "$0")")"
export PYTHONDONTWRITEBYTECODE=1
bash -n "$ROOT/verify.sh"
python3 -W error -m unittest discover -s "$ROOT" -p 'test_*.py' -v
python3 -W error "$ROOT/static_check.py"
printf '%s\n' 'ALL GREEN: source-only gate, fixed training-gradient diagnostic and unchanged qualification controls; neural work remains remote'
