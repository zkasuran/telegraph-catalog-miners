# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only augmentation contracts. No neural imports or generated fixtures."""
import json
from collections import Counter

from common import COUNTS, identity, lines, pair, sha
from development_contract import repetition_gate


def validate_plan(plan, sources, source_sha):
    if (plan["schema_version"] != 1 or plan["source_manifest_sha256"] != source_sha or
            plan["split"] != "train" or plan["audit_used"] is not False or
            not 80 <= len(plan["windows"]) <= 128 or plan["maximum_generations"] != 128):
        raise ValueError("augmentation plan source/split/bound mismatch")
    articles = {r["source_id"]: r for r in sources["source_report"]["articles"]["train"]}
    seen = set()
    for view in plan["windows"]:
        article = articles.get(view["source_id"])
        if (not article or view["split"] != "train" or view["group_id"] != article["group_id"] or
                view["year"] != article["year"] or view["source_sha256"] != article["sha256"]):
            raise ValueError("new window is not an original training source")
        prefix_start, prefix_end = view["prefix_span"]
        start, end = view["human_span"]
        if (article["paragraph"][prefix_start:prefix_end] != view["prefix"] or
                article["paragraph"][start:end] != view["human_source_window"] or
                prefix_start < article["human_span"][1] or start < prefix_end or
                len(view["human_source_window"].split()) != 24):
            raise ValueError("new source window span overlaps or mismatches original")
        if view["view_id"] in seen or view["seed"] != int(view["view_id"][:8], 16):
            raise ValueError("duplicate window or wrong generation seed")
        seen.add(view["view_id"])
    return articles


def validate_augmented(raw, plan, sources, report):
    if (sha(raw) != report["output_sha256"] or report["source_manifest_sha256"] != plan["source_manifest_sha256"] or
            report["generator"] != sources["generators"]["gpt2"]):
        raise ValueError("actual augmentation output/producer/source hash mismatch")
    rows = lines(raw)
    allowed = {v["view_id"]: v for v in plan["windows"]}
    seen = {}
    for row in rows:
        view = allowed.get(row.get("view_id"))
        if (not view or row["split"] != "train" or row["source_id"] != view["source_id"] or
                row["group_id"] != view["group_id"] or row["source_sha256"] != view["source_sha256"]):
            raise ValueError("augmentation changed source/entity/split role")
        if row["origin"] == "human_control":
            if not view["human_source_window"].startswith(row["text"]):
                raise ValueError("human augmentation text not a literal source prefix")
        elif row["origin"] == "synthetic":
            if (row.get("generator") != "gpt2" or row.get("generator_weight_sha256") != sources["generators"]["gpt2"]["weight_sha256"] or
                    row.get("seed") != view["seed"]):
                raise ValueError("new synthetic origin lacks actual pinned producer")
        else:
            raise ValueError("invalid augmented origin")
        if not 32 <= len(row["text"].encode()) <= 256:
            raise ValueError("augmented text violates literal byte ceiling")
        seen.setdefault(row["view_id"], []).append(row)
    if len(seen) < plan["minimum_valid_windows"] or any(len(rs) != 2 or {r["origin"] for r in rs} != {"human_control", "synthetic"} for rs in seen.values()):
        raise ValueError("insufficient or unbalanced actual augmented views")
    if len(seen) != report["valid_views"]:
        raise ValueError("actual augmented source-view count differs from report")
    return rows


def expanded_training_pairs(original, augmented, bare_group_count=4):
    """Keep whole views and article groups; guarantee reuse bound for any split."""
    view_order = list(dict.fromkeys(r["view_id"] for r in augmented))
    # Every optimizer accumulation block is complete. Trim at most three whole
    # generated views in fixed plan order, not by label, score or model outcome.
    kept = set(view_order[:len(view_order) - len(view_order) % 4])
    added = [r for r in augmented if r["view_id"] in kept]
    if len(kept) < 80:
        raise ValueError("insufficient structurally valid train-only views")
    raw = [r for r in original if r["split"] == "train"] + added
    if not 1 <= bare_group_count <= 6:
        raise ValueError("bounded original bare article exposure required")
    bare_groups = set(sorted({r["group_id"] for r in original if r["split"] == "train"})[:bare_group_count])
    pairs = []
    for row in raw:
        is_original = "view_id" not in row
        style = "bare_label" if is_original and row["group_id"] in bare_groups else "json" if int(row["group_id"][:2], 16) % 2 == 0 else "prose"
        current = pair(row, style)
        current["view_id"] = row.get("view_id", "original:" + row["group_id"])
        current["layout"] = "verdict_first"
        # A single symmetric layout distribution across old and new contexts.
        layout_id = row.get("view_id", row["group_id"])
        if style != "bare_label" and int(layout_id[2:4], 16) % 2:
            if style == "json":
                for side in ("chosen", "rejected"):
                    parsed = json.loads(current[side])
                    current[side] = json.dumps({"cues": parsed["cues"], "verdict": parsed["verdict"]}, ensure_ascii=False, separators=(",", ":"))
            else:
                text = row["text"]
                good = current["label"]
                bad = "AI-generated" if good == "Human" else "Human"
                current["chosen"] = f"Text assessed: {text}. Verdict: {good}"
                current["rejected"] = f"Text assessed: {text}. Verdict: {bad}"
            current["layout"] = "verdict_last"
        pairs.append(current)
    if len(pairs) % 8:
        raise ValueError("incomplete accumulation block")
    counted = repetition_gate(pairs)
    counts = Counter(r["rejected"].strip() for r in pairs)
    max_possible_leak = sum(n - 1 for n in counts.values()) / counted["internal_holdout"]
    if not counted["passed"] or max_possible_leak > 0.10:
        raise ValueError("expanded corpus violates unchanged repetition/leak gates")
    return pairs, {"original_train_pairs": 256, "new_train_pairs": len(added), "total_train_pairs": len(pairs),
                   "independent_article_groups": len({r["group_id"] for r in pairs}), "valid_added_views": len(kept),
                   "kept_view_ids": sorted(kept),
                   "trimmed_views": len(view_order) - len(kept), "bare_pairs": sum(r["style"] == "bare_label" for r in pairs),
                   "actual_repetition": counted, "worst_case_internal_negative_leak": max_possible_leak,
                   "gate_thresholds_changed": False, "validation_or_audit_used_for_fitting": False}
