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
# Match the remote natural entrypoint's cold import context, not this gate's
# PYTHONPATH. This loads checked source only and never prepares a model.
python3 -I -W error "$here/natural/bridge.py" --source-smoke
python3 -W error "$here/candidate/check_assets.py"
python3 -W error "$here/natural/verify_source.py"
# Source verification includes both dataset gates, frozen source-confidence
# scheduling and all retained train-fitted baselines. No neural entrypoint runs.
