# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Retain emitted source-origin rows and enforce a bounded same-seed extension."""
from common import sha, lines, audit_excerpts, identity
from secondary_contract import expanded_source_bundle

PREFIX_SHA = "560aa8bc4a15593b1819f8fe06e05a53a0f9851ddf56edc235c9f593bfff438d"
PRODUCER_COMMIT = "4099c81a0154f8ec12e2ad8acd3732052f5d806b"


def generation_policy(recipe):
    expected = {"retained_sha256": PREFIX_SHA, "retained_groups": 39, "original_token_budget": 64,
        "remaining_token_budget": 128, "same_seed_prefix_control": True, "source_selection_changed": False,
        "word_gate_changed": False, "audit_used": False}
    spec = recipe.get("secondary_generation_resume")
    if spec != expected or spec["same_seed_prefix_control"] is not True or any(spec[k] is not False for k in
            ("source_selection_changed", "word_gate_changed", "audit_used")) or any(type(spec[k]) is not int for k in
            ("retained_groups", "original_token_budget", "remaining_token_budget")):
        raise ValueError("fixed retained-prefix generation policy required")
    return dict(expected)


def retained_prefix(raw, provenance):
    bundle, secondary, _, base = expanded_source_bundle()
    if (sha(raw) != PREFIX_SHA or provenance.get("data_sha256") != PREFIX_SHA or
            provenance.get("run_id") != 44784633424 or provenance.get("commit") != PRODUCER_COMMIT or
            provenance.get("source_bundle_sha256") != bundle or provenance.get("audit_used") is not False):
        raise ValueError("retained secondary prefix lost original byte/producer/source binding")
    records = lines(raw)
    if not raw.endswith(b"\n") or not records or len(records) % 2 or len(records) >= 128:
        raise ValueError("retained prefix is not a complete partial source-pair stream")
    seen = set()
    for index in range(len(records) // 2):
        source = secondary["articles"][index]
        human, machine = records[2 * index:2 * index + 2]
        for row in (human, machine):
            if (row["source_id"] != source["source_id"] or row["group_id"] != source["group_id"] or
                    row["year"] != source["year"] or row["split"] != "train" or row["source_sha256"] != source["sha256"] or
                    row["licence"] != source["licence"] or row["source_attribution"] != source["url"]):
                raise ValueError("retained prefix changed a source role or original grant")
        if (human["origin"] != "human_control" or machine["origin"] != "synthetic" or machine["generator"] != "gpt2" or
                machine["generator_weight_sha256"] != base["generators"]["gpt2"]["weight_sha256"] or
                machine["generator_revision"] != base["generators"]["gpt2"]["revision"] or
                machine["seed"] != int(source["group_id"][:8], 16) or len(machine["continuation_token_ids"]) != 64):
            raise ValueError("retained prefix changed actual generator identity or seed")
        h, m, crop = audit_excerpts(source["human_text"], machine["decoded_continuation"])
        if h != human["text"] or m != machine["text"] or human["excerpt_view"] != crop or machine["excerpt_view"] != crop:
            raise ValueError("retained prefix changed exact source excerpt construction")
        for row in (human, machine):
            mark = identity(row["text"])
            if mark in seen:
                raise ValueError("retained prefix normalized duplicate")
            seen.add(mark)
    if len(records) != provenance.get("records"):
        raise ValueError("retained prefix observed count changed")
    return records


def check_token_prefix(short, extended):
    # Explicit token-array contract only. This function does not run a model.
    if len(short) != 64 or len(extended) != 128 or any(type(v) is not int or not 0 <= v < 50257 for v in short + extended):
        raise ValueError("bounded GPT-2 token prefix contract failed")
    if extended[:64] != short:
        raise ValueError("same-seed longer generation changed its observed first64 tokens")
