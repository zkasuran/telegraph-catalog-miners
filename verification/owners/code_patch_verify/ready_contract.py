# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Source-only report wire validation. Shape is not audit or producer proof."""
import json
import math
import re

FAMILY = "CODE_PATCH_VERIFY"
FOUNDATION = "0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6"
ENGINE = "77643d6e1176ab8744801cef93ccb844d0136ed32b8daa865d541f37d13292de"
TOKENIZER = "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456"
MAX_REPORT_BYTES = 1 << 20
MAX_DEPTH = 64


def decode(raw):
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_REPORT_BYTES:
        raise ValueError("bounded report bytes required")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)

    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite JSON number")
        return result

    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant, parse_float=number)
    except (UnicodeDecodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError("invalid report JSON") from error
    pending = [(result, 0)]
    while pending:
        item, depth = pending.pop()
        if isinstance(item, (dict, list)):
            if depth >= MAX_DEPTH:
                raise ValueError("report depth ceiling")
            children = item.values() if isinstance(item, dict) else item
            pending.extend((child, depth + 1) for child in children)
    if not isinstance(result, dict):
        raise ValueError("report must be an object")
    return result


def _hash(value, size):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{" + str(size) + "}", value) is not None


def _count(value, minimum):
    return type(value) is int and value >= minimum


def _metric(value):
    if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
        raise ValueError("finite numeric accuracy required")
    return value


def _names(values):
    if (not isinstance(values, tuple) or not 1 <= len(values) <= 8
            or not all(isinstance(value, str) and 0 < len(value) <= 80 for value in values)
            or len(set(values)) != len(values)):
        raise ValueError("frozen literal format/label names required")
    return set(values)


def validate_wire_shape(raw, expected, claimed_formats, labels):
    """Require separately measured identities. Never infer or authenticate them."""
    report = decode(raw)
    formats_expected, labels_expected = _names(claimed_formats), _names(labels)
    if (type(report.get("schema_version")) is not int or report["schema_version"] != 1
            or report.get("status") != "REGISTRATION_READY" or report.get("family") != FAMILY):
        raise ValueError("literal ready identity required")
    fields = {"run_id", "commit", "package_sha256", "package_keccak", "source_manifest_sha256", "serving_commit"}
    if not isinstance(expected, dict) or set(expected) != fields:
        raise ValueError("complete independent expected identity binding required")
    if not _count(expected["run_id"], 1):
        raise ValueError("actual integer run identity required")
    for key in ("commit", "serving_commit"):
        if not _hash(expected[key], 40):
            raise ValueError("literal source commit required")
    for key in ("package_sha256", "package_keccak", "source_manifest_sha256"):
        if not _hash(expected[key], 64):
            raise ValueError("literal lowercase bare hash required")
    for key in fields - {"source_manifest_sha256"}:
        if type(report.get(key)) is not type(expected[key]) or report[key] != expected[key]:
            raise ValueError("report identity differs: " + key)
    for key, value in (("foundation_sha256", FOUNDATION), ("engine_sha256", ENGINE), ("tokenizer_sha256", TOKENIZER)):
        if report.get(key) != value:
            raise ValueError("pinned runtime identity differs")
    source = report.get("source")
    if (not isinstance(source, dict) or type(source.get("normalized_overlap")) is not int
            or source["normalized_overlap"] != 0 or source.get("audit_used_for_selection") is not False
            or source.get("grants_verified") is not True or not _count(source.get("independent_groups"), 32)
            or source.get("source_manifest_sha256") != expected["source_manifest_sha256"]):
        raise ValueError("independent source and audit boundary required")
    quality = report.get("quality")
    if (not isinstance(quality, dict) or quality.get("passed") is not True
            or not _count(quality.get("audit_pairs"), 64) or quality.get("all_supported_formats_tested") is not True
            or not isinstance(quality.get("formats"), dict) or set(quality["formats"]) != formats_expected):
        raise ValueError("complete actual format audit required")
    for metrics in quality["formats"].values():
        if not isinstance(metrics, dict) or not isinstance(metrics.get("per_label"), dict) or set(metrics["per_label"]) != labels_expected:
            raise ValueError("complete label coverage required")
        accuracy = _metric(metrics.get("accuracy"))
        untrained, shortcut = _metric(metrics.get("untrained_accuracy")), _metric(metrics.get("shortcut_accuracy"))
        if (accuracy < 0.80 or min(_metric(v) for v in metrics["per_label"].values()) < 0.70
                or accuracy + 1e-9 < untrained + 0.05 or accuracy + 1e-9 < shortcut + 0.10):
            raise ValueError("unchanged quality and shortcut gates failed")
    runtime = report.get("runtime")
    if (not isinstance(runtime, dict) or any(runtime.get(key) is not True for key in
                                           ("full_parser_passed", "reference_engine_passed", "repeat_bits_identical"))
            or not _count(runtime.get("negative_identity_controls"), 3)
            or not isinstance(runtime.get("vectors"), list) or not 8 <= len(runtime["vectors"]) <= 100):
        raise ValueError("real reference parser/runtime controls required")
    identities = set()
    for vector in runtime["vectors"]:
        if not isinstance(vector, dict) or set(vector) != {"tokens", "score_bits"}:
            raise ValueError("exact reference vector wire fields required")
        tokens, bits = vector["tokens"], vector["score_bits"]
        if (not isinstance(tokens, list) or not 1 <= len(tokens) <= 512
                or not all(type(token) is int and 0 <= token < 50368 for token in tokens)
                or type(bits) is not int or not 0 <= bits < 1 << 32 or bits & 0x7f800000 == 0x7f800000):
            raise ValueError("invalid tokens or finite score bits")
        identity = tuple(tokens)
        if identity in identities:
            raise ValueError("repeated vector does not add independent fixed coverage")
        identities.add(identity)
    return report
