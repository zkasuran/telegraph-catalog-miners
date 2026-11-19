#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
set -euo pipefail
ROOT="$(dirname "$(realpath "$0")")"
PYTHONDONTWRITEBYTECODE=1 python3 -W error "$ROOT/verify.py" "$@"
