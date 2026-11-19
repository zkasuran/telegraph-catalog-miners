# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""A failed or incomplete validation result must not unlock an independent audit."""
from contracts import FORMATS, LABELS, quality


def require_validation(formats, original_pairs):
    if type(original_pairs) is not int or original_pairs < 16:
        raise ValueError("at least 16 original development pairs are required")
    if not isinstance(formats, dict) or set(formats) != set(FORMATS):
        raise ValueError("every claimed response format must be validated before audit")
    for name, metrics in formats.items():
        if (metrics.get("original_pairs") != original_pairs or set(metrics.get("per_label", {})) != set(LABELS)
                or metrics.get("fit_split") != "validation"):
            raise ValueError("validation identity or per-label coverage is incomplete: "+name)
    verdict = quality(formats)
    if not verdict["passed"]:
        raise ValueError("validation quality gate failed; independent audit remains locked")
    return {"schema_version": 1, "status": "VALIDATION_READY", "formats": formats,
            "original_pairs": original_pairs, "audit_used_for_selection": False}


def warmstart_policy():
    # Chosen from the failed training/development trace, not audit error rows.
    return {"epochs": 16, "learning_rate": 0.0002, "patience": 6, "train_formats": list(FORMATS),
            "initialization": "resume the retained licensed best checkpoint", "max_tokens": 384,
            "batch_size": 2, "grad_accum": 4, "rank": 8, "alpha": 32,
            "quality_gate": "unchanged 80% overall, 70% per label, +5 untrained and +10 strongest shortcut",
            "audit_policy": "do not unlock audit until all validation formats pass"}
