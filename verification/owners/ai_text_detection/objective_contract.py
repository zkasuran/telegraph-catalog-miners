# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pure train-only objective policy and scalar formula controls, no inference."""
import math

from training_weights import weight_policy


def objective_policy(recipe, rows=None):
    obj = recipe.get("correctness_auxiliary")
    expected = {"coefficient": 0.5, "chosen_target": 0.9, "rejected_target": 0.1,
                "scope": "existing_training_pair_sides", "reduction": "mean"}
    if obj != expected or any(type(obj[k]) not in (int, float) for k in
                              ("coefficient", "chosen_target", "rejected_target")):
        raise ValueError("fixed symmetric training-only correctness objective required")
    report = {**obj, "pair_loss_retained": True, "origin_specific_weights": False,
              "extra_training_rows": 0, "validation_audit_inputs_unchanged": True,
              "runtime_score_transform": False, "probability_claim": False}
    if rows is not None:
        _, weights = weight_policy(rows, recipe["bare_loss_weight"])
        for row in rows:
            if row["label"] != ("Human" if row["origin"] == "human_control" else "AI-generated"):
                raise ValueError("correctness targets require original origin preference labels")
            if row["chosen"] == row["rejected"] or row["margin"] != 0.0:
                raise ValueError("correctness objective requires distinct zero-margin source pairs")
        report["rows"] = len(rows)
        report["weight_by_origin"] = weights["weight_by_origin"]
    return report


def scalar_losses(chosen, rejected, policy):
    """Mathematical control on supplied scalars, not a model score calculation."""
    if any(not math.isfinite(x) for x in (chosen, rejected)):
        raise ValueError("finite scalar controls required")
    def softplus(x):
        return max(x, 0.0) + math.log1p(math.exp(-abs(x)))
    pair = softplus(rejected - chosen)
    point = (softplus(chosen) - chosen * policy["chosen_target"] +
             softplus(rejected) - rejected * policy["rejected_target"]) / 2
    return pair, point, pair + policy["coefficient"] * point
