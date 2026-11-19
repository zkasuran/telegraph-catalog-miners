#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
here="$(dirname "${BASH_SOURCE[0]}")"
export OWNER_SOURCE_ONLY=1
# Keep an externally installed guard first. The standalone checkout supplies
# the same barrier when it is run without the publication adapter.
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$here"
python3 -W error "$here/verify.py"
