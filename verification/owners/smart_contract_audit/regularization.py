# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Frozen training-only regularization choice, separate from quality gates."""
import hashlib
import json
import math
from pathlib import Path

SETTINGS = "regularization-v1.json"


def load(path):
    raw = Path(path).read_bytes()
    if len(raw) > 4096:
        raise ValueError("regularization config ceiling")
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"schema_version", "SPDX-License-Identifier", "method", "lora_dropout", "selection_data", "quality_gates_changed"}:
        raise ValueError("closed regularization schema required")
    dropout = value["lora_dropout"]
    if value["schema_version"] != 1 or value["SPDX-License-Identifier"] != "LicenseRef-zkasuran-SAND-1.0" or value["selection_data"] != "training/validation only" or value["quality_gates_changed"] is not False:
        raise ValueError("training-only regularization record required")
    if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("finite bounded LoRA dropout required")
    return dropout, hashlib.sha256(raw).hexdigest()
