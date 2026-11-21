# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-conditioned factual calibration plus ranking, training only."""
import hashlib
import json
import math
from pathlib import Path

SETTINGS = "objective-v1.json"


def load(path):
    raw = Path(path).read_bytes()
    if len(raw) > 4096:
        raise ValueError("training objective config ceiling")
    value = json.loads(raw)
    required = {"schema_version", "SPDX-License-Identifier", "method", "ranking_numerator", "calibration_numerator", "denominator", "target_source", "audit_used_for_selection", "quality_gates_changed"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("closed training objective schema required")
    if value["schema_version"] != 1 or value["SPDX-License-Identifier"] != "LicenseRef-zkasuran-SAND-1.0" or value["target_source"] != "training EVM-supported contradictory preferences only" or value["audit_used_for_selection"] is not False or value["quality_gates_changed"] is not False:
        raise ValueError("training-only objective record required")
    if any(type(value[k]) is not int for k in ("ranking_numerator", "calibration_numerator", "denominator")) or (value["ranking_numerator"], value["calibration_numerator"], value["denominator"]) != (1, 1, 2):
        raise ValueError("frozen normalized ranking/calibration objective required")
    return value, hashlib.sha256(raw).hexdigest()


def softplus(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("finite diagnostic scalar required")
    return max(value, 0) + math.log1p(math.exp(-abs(value)))


def scalar_loss(chosen, rejected, margin=0):
    ranking = softplus(-(chosen - rejected - margin))
    calibration = (softplus(-chosen) + softplus(rejected)) / 2
    return (ranking + calibration) / 2
