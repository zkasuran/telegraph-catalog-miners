# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""A separate additive training block, preserving every old source/producer hash."""
import json
from collections import Counter

from common import HERE, sha, lines, identity, pair
from compact_contract import compact_training
from origin_contract import equal_byte_views
from supplement_contract import source_bundle, validate_sources
from development_contract import repetition_gate


def expanded_source_bundle():
    parent, first, base = source_bundle()
    raw = (HERE / "expanded-training-source-bundle.json").read_bytes()
    binding = json.loads(raw)
    secondary_raw = (HERE / "secondary-training-sources.json").read_bytes()
    secondary = json.loads(secondary_raw)
    if (binding != {"schema_version": 1, "family": "AI_TEXT_DETECTION", "parent_file": "training-source-bundle.json",
            "parent_sha256": parent, "secondary_file": "secondary-training-sources.json", "secondary_sha256": sha(secondary_raw)} or
            secondary.get("parent_source_bundle_sha256") != parent or secondary.get("scope") != "training_only"):
        raise ValueError("expanded source bundle changed its original or secondary identities")
    overlay = json.loads(json.dumps(base))
    overlay["source_report"]["articles"]["train"] += first["articles"]
    validate_sources(secondary, overlay)
    return sha(raw), secondary, first, base


def attributions(rows):
    return [{k: r[k] for k in ("source_id", "authors", "url", "licence", "licence_url", "grant_clause", "changes", "sha256")} for r in rows]


def validate_secondary(raw, report, originals):
    bundle, secondary, _, base = expanded_source_bundle()
    if (report.get("output_sha256") != sha(raw) or report.get("source_bundle_sha256") != bundle or
            report.get("secondary_source_manifest_sha256") != sha((HERE / "secondary-training-sources.json").read_bytes()) or
            report.get("audit_used") is not False):
        raise ValueError("secondary generation lost its byte/source/audit binding")
    if "retained_sha256" in report:
        from generation_resume import PREFIX_SHA
        prefix = (HERE / "recovery/secondary-prefix.jsonl").read_bytes()
        if (sha(prefix) != PREFIX_SHA or report["retained_sha256"] != PREFIX_SHA or
                report.get("retained_records") != 78 or report.get("retained_groups") != 39 or
                report.get("retained_producer_run_id") != 44784633424 or
                report.get("remaining_generation_token_budget") != 128 or report.get("same_job_prefix_controls") != 25 or
                not raw.startswith(prefix)):
            raise ValueError("secondary completion changed retained bytes or fixed prefix controls")
    articles = {r["source_id"]: r for r in secondary["articles"]}
    rows = []
    for row in lines(raw):
        source = articles.get(row["source_id"])
        if (source is None or row["split"] != "train" or row["year"] != source["year"] or
                row["group_id"] != source["group_id"] or row["source_sha256"] != source["sha256"] or
                not 32 <= len(row["text"].encode()) <= 256):
            raise ValueError("secondary generation changed original source roles")
        if row["origin"] == "human_control":
            if not source["human_text"].startswith(row["text"]):
                raise ValueError("secondary human control is not a source prefix")
        elif row["origin"] == "synthetic":
            if row.get("generator") != "gpt2" or row.get("generator_weight_sha256") != base["generators"]["gpt2"]["weight_sha256"]:
                raise ValueError("secondary synthetic provenance changed")
        else:
            raise ValueError("secondary source origin missing")
        rows.append(pair({**row, "view_id": "secondary:" + row["group_id"]}))
    if len(rows) != 128 or len({r["group_id"] for r in rows}) != 64:
        raise ValueError("secondary actual generation incomplete")
    views = equal_byte_views(rows)
    new_ids = {r["text_id"] for r in views} | {r["text_id"] for r in rows}
    old_ids = {r["text_id"] for r in originals} | {r.get("original_text_id", r["text_id"]) for r in originals}
    if len({r["text_id"] for r in views}) != 128 or len({r["text_id"] for r in rows}) != 128 or new_ids.intersection(old_ids):
        raise ValueError("secondary normalized source overlap")
    return views


def secondary_preferences(views, original_preferences, bare_groups=0):
    if len(views) != 128 or len(original_preferences) != 880 or any(r["split"] != "train" for r in views):
        raise ValueError("secondary preference roles/count changed")
    if type(bare_groups) is not int or bare_groups not in (0, 2):
        raise ValueError("secondary minimal-verdict exposure is fixed to zero or two whole groups")
    selected = set(sorted({v["group_id"] for v in views})[:bare_groups])
    result = compact_training([pair({**v, "origin": "human_control" if v["human"] else "synthetic",
        "generator": "publisher" if v["human"] else "gpt2"},
        "bare_label" if v["group_id"] in selected else "json" if int(v["group_id"][:2], 16) % 2 == 0 else "prose") for v in views])
    if {r["group_id"] for r in result}.intersection(r["group_id"] for r in original_preferences):
        raise ValueError("secondary preference source groups overlap")
    combined = original_preferences + result
    repetition = repetition_gate(combined)
    duplicates = sum(n - 1 for n in Counter(r["rejected"].strip() for r in combined).values())
    worst_leak = duplicates / repetition["internal_holdout"]
    if not repetition["passed"] or len({r["group_id"] for r in combined}) != 256:
        raise ValueError("secondary preferences failed unchanged repetition/group gate")
    if worst_leak > 0.10:
        raise ValueError("secondary minimal-verdict exposure violates unchanged worst-case negative-leak gate")
    return result, {"original_preferences": 880, "added_preferences": 128, "combined_preferences": 1008,
        "independent_training_groups": 256, "new_bare_rows": 2 * bare_groups, "normalized_overlap": 0,
        "bare_group_selector": "lowest source-group hashes", "bare_group_ids": sorted(selected),
        "worst_case_negative_leak": worst_leak,
        "repetition": repetition, "audit_used": False,
        "preference_rows_sha256": sha(json.dumps(result, sort_keys=True, ensure_ascii=False).encode())}
