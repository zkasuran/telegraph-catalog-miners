# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Primary trainer pair-loader contract, preserving the literal scored strings."""
import math


def trainer_pairs(rows):
    result = []
    for row in rows:
        for field in ("prompt", "chosen", "rejected"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError("trainer requires complete literal preference strings")
        margin = row.get("margin", 0.0)
        if type(margin) not in {int, float} or not math.isfinite(margin):
            raise ValueError("trainer margin must be finite numeric input")
        # Primary read_pairs_jsonl uses _pair(..., margin=0.0). Metadata remains
        # source-verified outside the neural input, not stripped from artifacts.
        result.append({"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"], "margin": float(margin)})
    return result
