# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pure training-only source-margin policy and arithmetic controls, not inference."""
import math


def source_margin_policy(recipe):
    expected = {"coefficient": 0.1, "beta": 1.0, "target_min": 0.25, "target_max": 6.0,
                "teacher_detached": True, "source_labels_unchanged": True,
                "scope": "existing_training_rows", "runtime_transform": False, "audit_used": False}
    spec = recipe.get("source_margin_consistency")
    if (spec != expected or any(type(spec[k]) not in (int, float) for k in ("coefficient", "beta", "target_min", "target_max")) or
            spec["teacher_detached"] is not True or spec["source_labels_unchanged"] is not True or
            spec["runtime_transform"] is not False or spec["audit_used"] is not False):
        raise ValueError("fixed detached source-margin consistency required")
    return dict(expected)


def scalar_target(logit, human, policy):
    if not math.isfinite(logit) or type(human) is not int or human not in (0, 1):
        raise ValueError("finite source scalar and recorded binary origin required")
    x = (2 * human - 1) * logit
    softplus = x if x > 20 else max(x, 0) + math.log1p(math.exp(-abs(x)))
    return min(policy["target_max"], max(policy["target_min"], 2 * softplus))


def scalar_consistency(chosen, rejected, logit, human, policy):
    if any(not math.isfinite(v) for v in (chosen, rejected)):
        raise ValueError("finite student scalar controls required")
    gap = abs(chosen - rejected - scalar_target(logit, human, policy))
    return 0.5 * gap * gap if gap < 1 else gap - 0.5
