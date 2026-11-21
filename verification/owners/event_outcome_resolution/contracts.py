# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Closed public ready-report contract with no model or credential imports."""
import math
import re

from common import FOUNDATION, DEFAULT_OWNER
from parent_scope import ready_parent_material, verify_scope

FAMILY = "EVENT_OUTCOME_RESOLUTION"
FORMATS = ("json", "prose")
SHA = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")


def finite_bits(value):
    return type(value) is int and 0 <= value < (1 << 32) and value & 0x7f800000 != 0x7f800000


def object_field(value, name):
    if not isinstance(value, dict):
        raise ValueError(name + " must be a JSON object")
    return value


def matches(pattern, value):
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def validate_quality_runtime(report):
    report = object_field(report, "measured audit evidence")
    quality = object_field(report.get("quality"), "quality evidence")
    formats = object_field(quality.get("formats"), "format metrics")
    if (quality.get("passed") is not True or type(quality.get("audit_pairs")) is not int or quality["audit_pairs"] < 64 or
            quality.get("all_supported_formats_tested") is not True or set(formats) != set(FORMATS)):
        raise ValueError("all claimed formats require complete measured audit")
    for metrics in formats.values():
        metrics = object_field(metrics, "format metric")
        labels = object_field(metrics.get("per_label"), "per-label metrics")
        if set(labels) != {"yes", "no", "unresolved"}:
            raise ValueError("all outcome labels required")
        values = [metrics.get("accuracy"), metrics.get("untrained_accuracy"), metrics.get("shortcut_accuracy"), *labels.values()]
        if not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError("finite measured quality required")
        acc = metrics["accuracy"]
        if acc < 0.80 or min(labels.values()) < 0.70 or acc + 1e-9 < metrics["untrained_accuracy"] + 0.05 or acc + 1e-9 < metrics["shortcut_accuracy"] + 0.10:
            raise ValueError("quality gate failed")
    runtime = object_field(report.get("runtime"), "runtime evidence")
    if (any(runtime.get(k) is not True for k in ("full_parser_passed", "reference_engine_passed", "repeat_bits_identical")) or
            type(runtime.get("negative_identity_controls")) is not int or runtime["negative_identity_controls"] < 3):
        raise ValueError("actual parser/runtime/identity proof required")
    vectors = runtime.get("vectors", [])
    if not isinstance(vectors, list) or not 8 <= len(vectors) <= 100:
        raise ValueError("bounded real reference vectors required")
    for vector in vectors:
        vector = object_field(vector, "reference vector")
        tokens = vector.get("tokens")
        if (not isinstance(tokens, list) or not 1 <= len(tokens) <= 512 or
                not all(type(t) is int and 0 <= t < 50368 for t in tokens) or not finite_bits(vector.get("score_bits"))):
            raise ValueError("literal token/score-bit vector invalid")
    return report


def validate_ready(report):
    report = object_field(report, "ready report")
    if (type(report.get("schema_version")) is not int or report.get("schema_version") != 1 or report.get("status") != "REGISTRATION_READY" or
            report.get("family") != FAMILY or type(report.get("run_id")) is not int or report["run_id"] < 1 or
            not matches(COMMIT, report.get("commit")) or report.get("serving_commit") != report["commit"]):
        raise ValueError("ready run/family/commit mismatch")
    for key in ("package_sha256", "package_keccak"):
        if not matches(SHA, report.get(key)):
            raise ValueError("bare lowercase package hash required")
    for field, file in (("foundation_sha256", "foundation_lora_fp32.onnx"), ("engine_sha256", "foundation_engine.wasm"),
                        ("tokenizer_sha256", "tokenizer.json")):
        if report.get(field) != FOUNDATION[file]:
            raise ValueError("pinned runtime identity mismatch")
    source = object_field(report.get("source"), "source evidence")
    mode = verify_scope(report, report.get("scope", "general"))
    if mode == "parent_only":
        ready_parent_material(source.get("parent_material"))
    if (type(source.get("normalized_overlap")) not in (int, float) or source.get("normalized_overlap") != 0 or source.get("audit_used_for_selection") is not False or
            source.get("grants_verified") is not True or not matches(SHA, source.get("source_manifest_sha256")) or
            type(source.get("independent_groups")) is not int or source["independent_groups"] < 32):
        raise ValueError("source or independent audit boundary mismatch")
    validate_quality_runtime(report)
    if report.get("owner") != DEFAULT_OWNER:
        raise ValueError("approved Telegraph identity mismatch")
    return report
