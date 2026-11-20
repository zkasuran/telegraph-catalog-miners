# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pure training-side counterfactual sibling batching, no model imports."""
from collections import defaultdict
import argparse
import json
from pathlib import Path

from sampling import weights


def pairs(rows, frozen):
    row_weights = weights(rows, frozen)  # Strict frozen train-only identity.
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        if not isinstance(row.get("sibling_group"), str) or not row["sibling_group"] or type(row.get("vulnerable")) is not bool:
            raise ValueError("complete training sibling identity required")
        grouped[(row["sibling_group"], row["response_view"])].append(i)
    result, pair_weights = [], []
    for key in sorted(grouped):
        indices = grouped[key]
        if len(indices) != 2 or {rows[i]["vulnerable"] for i in indices} != {True, False}:
            raise ValueError("exact vulnerable/fixed sibling pair required")
        left, right = sorted(indices, key=lambda i: not rows[i]["vulnerable"])
        a, b = rows[left], rows[right]
        if a["template_group"] != b["template_group"] or row_weights[left] != row_weights[right]:
            raise ValueError("sibling implementation or weight differs")
        if a.get("facts") != b.get("facts") or a["source_sha256"] == b["source_sha256"]:
            raise ValueError("same call facts with distinct source siblings required")
        if not isinstance(a.get("chosen"), str) or not isinstance(a.get("rejected"), str) or a["chosen"] == a["rejected"]:
            raise ValueError("distinct literal preference strings required")
        if a["chosen"] != b.get("rejected") or a["rejected"] != b.get("chosen"):
            raise ValueError("literal sibling preferences must be reversed")
        result.append([left, right])
        pair_weights.append(row_weights[left])
    if len(result) * 2 != len(rows):
        raise ValueError("every training view must belong to exactly one sibling pair")
    return result, pair_weights


def report(rows, frozen):
    batches, coefficients = pairs(rows, frozen)
    return {"method": "weighted counterfactual sibling minibatches", "training_views": len(rows),
            "pair_batches": len(batches), "rows_per_batch": 2, "training_groups": len(frozen["coefficients"]),
            "pair_coefficients": sorted(set(coefficients)), "literal_preferences_reversed": True,
            "both_labels_per_batch": True, "audit_used_for_fitting": False,
            "source_rows_or_preferences_added": False,
            "limits": "Batch plan only, not actual sampling, model training or node qualification"}


def main():
    from data import response_views
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    with args.train.open("rb") as stream:
        raw = stream.read((2 << 20) + 1)
    if len(raw) > 2 << 20:
        raise ValueError("training source byte ceiling")
    rows = response_views([json.loads(line) for line in raw.decode().splitlines() if line.strip()], "train")
    result = report(rows, json.loads(args.frozen.read_text()))
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
