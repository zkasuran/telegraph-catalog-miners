# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Training-only verdict preferences on verified supplementary provenance.

Call validate_generation first. These constructed preferences are not human votes.
No source selection, generator operation or runtime transformation occurs here.
"""
import json

from common import identity, pair, sha
from compact_contract import compact_training
from development_contract import repetition_gate


def preference_policy(recipe):
    expected = {"scope": "training_only", "independent_groups": 64,
                "new_preference_rows": 128, "new_bare_rows": 0,
                "original_preference_rows": 752, "supplement_updates": 16,
                "pair_coefficient": 1.0, "original_phase_unchanged": True,
                "runtime_transform": False, "audit_used": False}
    spec = recipe.get("supplement_preferences")
    if (spec != expected or any(type(spec[k]) is not int for k in
            ("independent_groups", "new_preference_rows", "new_bare_rows", "original_preference_rows", "supplement_updates")) or
            type(spec["pair_coefficient"]) not in (int, float) or
            spec["original_phase_unchanged"] is not True or spec["runtime_transform"] is not False or spec["audit_used"] is not False):
        raise ValueError("fixed training-only supplementary preference policy required")
    return dict(expected)


def preference_rows(views, original, validation):
    if len(views) != 128 or len(original) != 752 or len(validation) != 64:
        raise ValueError("supplementary preference population changed")
    if any(r["split"] != "train" for r in original) or any(r["split"] != "validation" for r in validation):
        raise ValueError("supplementary preference split roles changed")
    old_groups = {r["group_id"] for r in original + validation}
    old_ids = {r["text_id"] for r in original + validation}
    grouped, seen = {}, set()
    pairs = []
    for view in views:
        if (view["split"] != "train" or type(view["year"]) is not int or not 2012 <= view["year"] <= 2014 or
                view["group_id"] != identity(view["source_id"]) or
                view["view_id"] != "supplement:" + view["group_id"] or
                view["group_id"] in old_groups or type(view["human"]) is not int or view["human"] not in (0, 1)):
            raise ValueError("supplementary preference source/role identity failed")
        text_id = identity(view["text"])
        if (view["text_id"] != text_id or view["text_bytes"] != len(view["text"].encode()) or
                not 32 <= view["text_bytes"] <= 256 or text_id in seen or
                {text_id, view["original_text_id"]}.intersection(old_ids)):
            raise ValueError("supplementary preference normalized/byte identity failed")
        seen.add(text_id)
        grouped.setdefault(view["view_id"], []).append(view)
        origin = "human_control" if view["human"] else "synthetic"
        style = "json" if int(view["group_id"][:2], 16) % 2 == 0 else "prose"
        pairs.append(pair({**view, "origin": origin, "generator": "publisher" if view["human"] else "gpt2"}, style))
    if len(grouped) != 64 or any(len(v) != 2 or {r["human"] for r in v} != {0, 1} or
                                len({r["text_bytes"] for r in v}) != 1 for v in grouped.values()):
        raise ValueError("supplementary preference paired equal-byte groups failed")
    result = compact_training(pairs)
    if any(r["style"] == "bare_label" for r in result):
        raise ValueError("supplementary preferences must not add repeated bare labels")
    combined = original + result
    if len({r["group_id"] for r in original}) != 128 or len({r["group_id"] for r in combined}) != 192:
        raise ValueError("supplementary preferences changed independent source counts")
    gate = repetition_gate(combined)
    if not gate["passed"]:
        raise ValueError("combined preference corpus failed unchanged repetition gate")
    binding = {"original_rows": 752, "supplementary_rows": 128, "combined_rows": len(combined),
               "independent_training_groups": len({r["group_id"] for r in combined}),
               "new_bare_rows": 0, "normalized_overlap": 0, "audit_used": False,
               "repetition": gate, "preference_rows_sha256": sha(json.dumps(result, sort_keys=True,
                    ensure_ascii=False, separators=(",", ":")).encode()),
               "preference_authorship": "Constructed from original source/GPT-2 provenance, not human-reviewed votes."}
    return result, binding
