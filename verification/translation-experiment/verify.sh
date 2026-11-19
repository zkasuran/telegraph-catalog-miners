#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
: "${SOURCE_PYTHON:?Set the existing source-only Python environment}"
: "${KIT:?Set the pinned reference kit directory}"
: "${TOKENIZER:?Set the released tokenizer file}"
: "${SOURCE_OUT:?Set an existing validated corpus cache directory}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1
python3 -W error -m unittest discover -s verification/translation-experiment -p 'test_*.py' -v
ruff check verification/translation-experiment
"$SOURCE_PYTHON" -W error verification/translation-experiment/pilot.py audit --out "$SOURCE_OUT" --kit "$KIT" --tokenizer "$TOKENIZER"
python3 -W error verification/translation-experiment/pilot.py final-gate
