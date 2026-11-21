# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Training-only grouped formats and authored scalar objective contracts."""
import math
import re

from contracts import FAMILY, FORMATS, LABELS
from data import normalized
from development_diagnostics import checked_pair

OBJECTIVES = ("independent", "paired-worst-format")
MAX_GROUP_BATCH = 8


def require_smoke(freeze, report, report_hash):
    objective = freeze.get("training_objective", "independent")
    if objective not in OBJECTIVES:
        raise ValueError("unknown frozen training objective")
    if objective == "independent":
        return
    if (not isinstance(report, dict) or report.get("schema_version") != 1
            or freeze.get("family") != FAMILY or report.get("family") != FAMILY
            or type(freeze.get("run_id")) is not int or freeze["run_id"] < 1 or report.get("run_id") != freeze["run_id"]
            or not isinstance(report_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", report_hash)
            or report.get("commit") != freeze.get("commit") or report.get("negative_controls") != 2
            or report.get("status") != "PAIRED_OBJECTIVE_TORCH_SMOKE_PASSED"
            or freeze.get("paired_objective_smoke_sha256") != report_hash):
        raise ValueError("actual paired objective smoke is not bound to this freeze")
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != 4:
        raise ValueError("complete actual objective controls required")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("actual objective control object required")
        gradients = case.get("actual_gradients")
        if (not isinstance(gradients, list) or len(gradients) != 2
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in gradients)):
            raise ValueError("finite actual objective gradients required")


def paired_training(rows):
    if not isinstance(rows, list) or not 2 <= len(rows) <= 5000:
        raise ValueError("bounded original training rows required")
    groups, seen = [], set()
    for row in rows:
        views = [{**checked_pair(row, style, "train"), "view_style": style} for style in FORMATS]
        key = normalized(row["evidence"])
        if key in seen:
            raise ValueError("duplicate normalized training evidence")
        seen.add(key)
        groups.append({**views[0], "paired_views": views})
    if {row["label"] for row in groups} != set(LABELS):
        raise ValueError("both original training source labels required")
    return groups


def flatten_grouped(batch):
    if not isinstance(batch, list) or not 1 <= len(batch) <= MAX_GROUP_BATCH:
        raise ValueError("bounded paired training batch required")
    flat = []
    for row in batch:
        if not isinstance(row, dict) or row.get("split") != "train":
            raise ValueError("paired objective refuses validation or audit rows")
        views = row.get("paired_views")
        if not isinstance(views, list) or len(views) != len(FORMATS):
            raise ValueError("exact grouped JSON/prose views required")
        for style, view in zip(FORMATS, views):
            expected = checked_pair(row, style, "train")
            if not isinstance(view, dict) or view.get("view_style") != style:
                raise ValueError("paired view order differs from the objective")
            if any(view.get(key) != expected[key] for key in ("prompt", "chosen", "rejected", "source_group", "evidence_sha256", "label", "split", "margin")):
                raise ValueError("paired source/view identity changed")
            flat.append(view)
    return flat


def grouped_collator(original_class):
    """Pure caller adapter; the supplied original tokenizer is never replaced."""
    class GroupedCollator:
        def __init__(self, tokenizer, max_length):
            self.original = original_class(tokenizer, max_length)

        def __call__(self, rows):
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ValueError("original preference row batch required")
            if any("paired_views" in row for row in rows):
                return self.original(flatten_grouped(rows))
            return self.original(rows)
    return GroupedCollator


def scalar_loss(json_margin, prose_margin, penalty=0.001):
    """Authored arithmetic reference only. Not a model or reference-engine score."""
    if (any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 10000 for x in (json_margin, prose_margin))
            or type(penalty) not in (int, float) or not math.isfinite(penalty) or not 0 <= penalty <= 1):
        raise ValueError("finite bounded authored margins and penalty required")
    weak = min(json_margin, prose_margin)
    bt = max(-weak, 0) + math.log1p(math.exp(-abs(weak)))
    return bt + penalty * (json_margin ** 2 + prose_margin ** 2) / 2
