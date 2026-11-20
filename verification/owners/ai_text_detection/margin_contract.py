# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pure fixed source-counterpart margin-alignment policy and scalar controls."""
import math


def margin_policy(recipe):
    expected = {"coefficient": 0.1, "beta": 1.0, "scope": "paired_training_verdict_margins", "runtime_transform": False}
    if recipe.get("margin_alignment") != expected:
        raise ValueError("fixed role-symmetric training margin policy required")
    return dict(expected)


def scalar_margin(a, b):
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("finite counterpart margin controls required")
    difference = abs(a - b)
    return 0.5 * difference * difference if difference < 1 else difference - 0.5
