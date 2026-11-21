# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Explicit parent-only plan, never a model-quality or registration-ready report."""
import argparse
import copy
import math
from pathlib import Path
import re

from common import FOUNDATION, public_json, sha, strict_json, write_json

FAMILY = "EVENT_OUTCOME_RESOLUTION"
MIN_CONTRIBUTORS = 5
SHA = re.compile(r"[a-f0-9]{64}")
REQUEST = [{"name": "event", "type": "string", "slot": "strings", "index": 0, "max_bytes": 256}]
RESPONSE = [{"name": "outcome", "type": "enum", "slot": "strings", "index": 0,
             "allowed": ["yes", "no", "unresolved"], "max_bytes": 256},
            {"name": "resolved_yes", "type": "bool", "slot": "bools", "index": 0}]


def object_value(value, label):
    if not isinstance(value, dict):
        raise ValueError(label + " must be an object")
    return value


def parent_contract(parent, resolver):
    parent = object_value(parent, "parent schema")
    resolver = object_value(resolver, "resolver")
    if (parent.get("intent_id") != FAMILY or parent.get("intent_name") != FAMILY or parent.get("class") != "judgment" or
            parent.get("prefix") is not True or resolver.get("intent") != FAMILY or resolver.get("class") != "judgment"):
        raise ValueError("exact judgment parent scope required, no class override")
    for section in ("request", "response", "onchain"):
        if object_value(parent.get(section), section).get("served_by") != FAMILY:
            raise ValueError("parent schema must be served by the exact parent")
    wire = parent["onchain"]
    for fields in (wire.get("request"), wire.get("response")):
        if not isinstance(fields, list) or not all(isinstance(item, dict) and type(item.get("index")) is int for item in fields):
            raise ValueError("literal wire indexes must be integers")
        if any("max_bytes" in item and type(item["max_bytes"]) is not int for item in fields):
            raise ValueError("literal wire byte bounds must be integers")
    if (type(wire.get("format_version")) is not int or wire["format_version"] != 4 or wire.get("ready") is not True or
            wire.get("request") != REQUEST or wire.get("response") != RESPONSE):
        raise ValueError("literal parent wire changed")
    return {"scope": "parent_only", "family": FAMILY, "class": "judgment", "claimed_intents": [FAMILY],
            "general_child_claimed": False, "max_tokens": 256, "request": copy.deepcopy(REQUEST), "response": copy.deepcopy(RESPONSE)}


def material_contract(status):
    status = object_value(status, "parent material")
    if status.get("intent_id") != FAMILY:
        raise ValueError("child material cannot qualify the parent")
    snapshot = object_value(status.get("active_snapshot"), "active snapshot")
    if not isinstance(snapshot.get("corpus_version"), str) or not SHA.fullmatch(snapshot["corpus_version"]):
        raise ValueError("active snapshot digest required")
    for field in ("eligible_submitters", "min_submitters"):
        if type(status.get(field)) is not int or status[field] < 0:
            raise ValueError("integer source contributor counts required")
    for field in ("submitters", "rows", "qualification", "ranking"):
        if type(snapshot.get(field)) is not int or snapshot[field] < 0:
            raise ValueError("integer snapshot counts required")
    if snapshot["qualification"] + snapshot["ranking"] != snapshot["rows"]:
        raise ValueError("snapshot split counts differ")
    required = max(MIN_CONTRIBUTORS, status["min_submitters"])
    ready = (status["eligible_submitters"] >= required and snapshot["submitters"] >= required and
             snapshot["qualification"] >= 10 and snapshot["ranking"] >= 10)
    return {"intent": FAMILY, "corpus_version": snapshot["corpus_version"], "required_contributors": required,
            "eligible_contributors": status["eligible_submitters"], "active_contributors": snapshot["submitters"],
            "material_ready": ready, "child_material_borrowed": False}


def training_record(validation_pairs, validation_accuracy):
    # Called only by a future remote export after actual validation. This plan
    # never calls it with expected values or substitutes an audit measurement.
    if (type(validation_pairs) is not int or validation_pairs < 1 or
            type(validation_accuracy) not in (int, float) or not math.isfinite(validation_accuracy) or
            not 0 <= validation_accuracy <= 1):
        raise ValueError("actual finite validation measurement required")
    return {"family": FAMILY, "children": [{"intent": FAMILY, "pairs": validation_pairs,
                                             "accuracy": validation_accuracy, "held_out": True}]}


def prepare(parent, resolver, status, frozen, frozen_sha):
    scope = parent_contract(parent, resolver)
    material = material_contract(status)
    frozen = object_value(frozen, "frozen source manifest")
    if not isinstance(frozen_sha, str) or not SHA.fullmatch(frozen_sha) or frozen.get("family") != FAMILY:
        raise ValueError("exact frozen source identity required")
    files = object_value(frozen.get("files"), "frozen splits")
    if (set(files) != {"train", "validation", "audit"} or type(frozen.get("cross_split_overlap")) not in (int, float) or
            frozen.get("cross_split_overlap") != 0):
        raise ValueError("grouped source split contract changed")
    for name, item in files.items():
        item = object_value(item, name)
        if (not isinstance(item.get("sha256"), str) or not SHA.fullmatch(item["sha256"]) or
                type(item.get("pairs")) is not int or item["pairs"] < 1):
            raise ValueError("frozen split identity/count required")
    if (type(files["audit"].get("independent_groups")) is not int or files["audit"]["independent_groups"] < 32 or
            files["audit"]["pairs"] < 64):
        raise ValueError("independent audit groups required")
    return {"schema_version": 1, "status": "PARENT_ONLY_SCOPE_PREPARED", **scope, "material": material,
            "source_manifest_sha256": frozen_sha, "frozen_split_sha256": {name: item["sha256"] for name, item in files.items()},
            "foundation": FOUNDATION, "training_record_intents": [FAMILY],
            "validation_accuracy": None, "training_record_emitted": False, "model_work_performed": False,
            "registration_ready": False, "audit_used_for_selection": False,
            "limits": ["A source-level parent-only plan is not full production caller or model qualification proof.",
                       "A real remote validation result is required before emitting a training record.",
                       "GENERAL remains excluded. No child contributor count or protected pool rows are used."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    parent, parent_raw = public_json("https://devnode.telegraphprotocol.com/engine/v1/intents/" + FAMILY)
    resolver, resolver_raw = public_json("https://devnode.telegraphprotocol.com/api/wasm?resolve=" + FAMILY)
    status, status_raw = public_json("https://devnode.telegraphprotocol.com/api/evalpool/status?intent_id=" + FAMILY)
    frozen_raw = Path(__file__).with_name("frozen").joinpath("frozen.json").read_bytes()
    result = prepare(parent, resolver["resolve"], status, strict_json(frozen_raw), sha(frozen_raw))
    result["source_response_sha256"] = {"parent": sha(parent_raw), "resolver": sha(resolver_raw), "material_status": sha(status_raw)}
    write_json(args.out, result)
    print(result["status"], "material_ready=" + str(result["material"]["material_ready"]))


if __name__ == "__main__":
    main()
