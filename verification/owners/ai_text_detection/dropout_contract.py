# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Explicit zero training-dropout policy, not a new model or score transform."""


def dropout_policy(recipe):
    expected = {"lora_dropout": 0.0, "scope": "training_only", "inference_contract_changed": False,
                "data_changed": False, "quality_gates_changed": False, "native_repeat_control_required": True}
    spec = recipe.get("comparison_dropout")
    if (spec != expected or type(spec["lora_dropout"]) not in (int, float) or
            any(spec[k] is not False for k in ("inference_contract_changed", "data_changed", "quality_gates_changed")) or
            spec["native_repeat_control_required"] is not True):
        raise ValueError("fixed zero comparison-dropout policy required")
    return dict(expected)
