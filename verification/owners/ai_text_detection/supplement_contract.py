# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Independent training-source supplement and composite provenance contracts."""
import json
import re
from common import HERE, identity, sha, lines
from origin_contract import equal_byte_views


def validate_sources(spec, base):
    if spec.get("schema_version") != 1 or spec.get("family") != "AI_TEXT_DETECTION" or len(spec.get("articles", [])) != 64:
        raise ValueError("64 independent supplementary training sources required")
    old = [r for rows in base["source_report"]["articles"].values() for r in rows]
    ids = {r["source_id"] for r in old} | set(base["audit_refresh"]["retired_article_ids"])
    authors = {a for r in old for a in r["author_ids"]} | set(base["audit_refresh"]["retired_author_ids"])
    texts = {identity(r[k]) for r in old for k in ("paragraph", "prefix", "human_text")}
    for r in spec["articles"]:
        marks = {identity(r[k]) for k in ("paragraph", "prefix", "human_text")}
        if (r["split"] != "train" or not 2012 <= r["year"] <= 2014 or r["group_id"] != identity(r["source_id"]) or
                r["source_id"] in ids or authors.intersection(r["author_ids"]) or texts.intersection(marks)):
            raise ValueError("supplementary source/entity/time/text separation failed")
        if (r["licence"] not in {"CC-BY-3.0", "CC-BY-4.0"} or "unrestricted" not in r["grant_clause"] or
                "credited" not in r["grant_clause"] or not r["authors"] or
                not re.fullmatch(r"[0-9a-f]{64}", r["sha256"]) or not re.fullmatch(r"[0-9a-f]{40}", r["git_blob"]) or
                not 0 < r["bytes"] <= 2 << 20 or not 32 <= len(r["human_text"].encode()) <= 256):
            raise ValueError("supplementary original grant or byte identity missing")
        a, b = r["human_span"]
        if r["paragraph"][a:b] != r["human_text"]:
            raise ValueError("supplementary human-source span changed")
        ids.add(r["source_id"])
        authors.update(r["author_ids"])
        texts.update(marks)
    return spec["articles"]


def source_bundle():
    raw = (HERE / "training-source-bundle.json").read_bytes()
    bundle = json.loads(raw)
    base_raw = (HERE / "sources.json").read_bytes()
    extra_raw = (HERE / "extra-training-sources.json").read_bytes()
    if (bundle.get("schema_version") != 1 or bundle.get("family") != "AI_TEXT_DETECTION" or
            bundle.get("base_file") != "sources.json" or bundle.get("extra_file") != "extra-training-sources.json" or
            bundle.get("base_sha256") != sha(base_raw) or bundle.get("extra_sha256") != sha(extra_raw)):
        raise ValueError("composite training-source hash does not bind both original manifests")
    base, extra = json.loads(base_raw), json.loads(extra_raw)
    validate_sources(extra, base)
    return sha(raw), extra, base


def validate_generation(raw, report, extra, base, original_views):
    if (report["output_sha256"] != sha(raw) or report["extra_source_manifest_sha256"] != sha((HERE / "extra-training-sources.json").read_bytes()) or
            report["source_bundle_sha256"] != source_bundle()[0] or report.get("audit_used") is not False):
        raise ValueError("supplementary actual generation byte/source binding failed")
    records = lines(raw)
    articles = {r["source_id"]: r for r in validate_sources(extra, base)}
    views = []
    for r in records:
        a = articles.get(r["source_id"])
        if (a is None or r["split"] != "train" or r["year"] != a["year"] or r["group_id"] != a["group_id"] or
                r["source_sha256"] != a["sha256"] or not 32 <= len(r["text"].encode()) <= 256):
            raise ValueError("supplementary generated record lost original source role")
        if r["origin"] == "human_control":
            if not a["human_text"].startswith(r["text"]):
                raise ValueError("supplementary human record is not an original source prefix")
        elif r["origin"] == "synthetic":
            if r["generator"] != "gpt2" or r["generator_weight_sha256"] != base["generators"]["gpt2"]["weight_sha256"]:
                raise ValueError("supplementary machine provenance mismatch")
        else:
            raise ValueError("supplementary origin annotation missing")
        views.append({"prompt": json.dumps({"text": r["text"]}, ensure_ascii=False, separators=(",", ":")),
            "style": "json", "source_id": r["source_id"], "group_id": r["group_id"], "view_id": "supplement:" + r["group_id"],
            "split": "train", "year": r["year"], "origin": r["origin"], "text_id": identity(r["text"]),
            "label": "Human" if r["origin"] == "human_control" else "AI-generated"})
    if len(views) != 128 or len({r["group_id"] for r in views}) != 64:
        raise ValueError("actual supplementary generation incomplete")
    cropped = equal_byte_views(views)
    new_ids = {r["text_id"] for r in cropped}
    raw_ids = {r["text_id"] for r in views}
    old_ids = {r["text_id"] for r in original_views} | {r.get("original_text_id", r["text_id"]) for r in original_views}
    if len(raw_ids) != len(views) or len(new_ids) != len(cropped) or (new_ids | raw_ids).intersection(old_ids):
        raise ValueError("supplementary normalized source-origin view overlap")
    return cropped
