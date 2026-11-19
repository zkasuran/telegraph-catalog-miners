# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Fail-closed registration-ready validation. Pure contracts, no neural imports."""
import math
import re

FOUNDATION = "0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6"
ENGINE = "77643d6e1176ab8744801cef93ccb844d0136ed32b8daa865d541f37d13292de"
TOKENIZER = "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456"
FORMATS = {"plain", "array", "object"}


def validate_ready(report):
    if report.get("schema_version") != 1 or report.get("status") != "REGISTRATION_READY" or report.get("family") != "KEYWORD_EXTRACTION":
        raise ValueError("not an exact-family registration-ready report")
    if type(report.get("run_id")) is not int or report["run_id"] < 1:
        raise ValueError("actual remote run ID required")
    if not re.fullmatch(r"[a-f0-9]{40}", report.get("commit", "")) or report.get("serving_commit") != report["commit"]:
        raise ValueError("audit and identical serving source commit required")
    for field in ("package_sha256", "package_keccak"):
        if not re.fullmatch(r"[a-f0-9]{64}", report.get(field, "")):
            raise ValueError("literal bare package hashes required")
    for field, expected in (("foundation_sha256", FOUNDATION), ("engine_sha256", ENGINE), ("tokenizer_sha256", TOKENIZER)):
        if report.get(field) != expected:
            raise ValueError("pinned actual runtime asset identity required")
    source = report.get("source", {})
    if source.get("normalized_overlap") != 0 or source.get("audit_used_for_selection") is not False or source.get("grants_verified") is not True:
        raise ValueError("source grants and independent audit boundary not established")
    if source.get("independent_groups", 0) < 32 or not re.fullmatch(r"[a-f0-9]{64}", source.get("source_manifest_sha256", "")):
        raise ValueError("bound independent source groups required")
    quality = report.get("quality", {})
    if quality.get("passed") is not True or quality.get("audit_pairs", 0) < 64 or quality.get("all_supported_formats_tested") is not True:
        raise ValueError("completed full independent quality audit required")
    formats = quality.get("formats", {})
    if set(formats) != FORMATS:
        raise ValueError("every exact claimed format must be tested")
    for name, m in formats.items():
        labels = m.get("per_label", {})
        values = [m.get("accuracy"), m.get("untrained_accuracy"), m.get("shortcut_accuracy"), *labels.values()]
        if not labels or any(type(v) not in {int, float} or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError(name + " missing finite actual metrics")
        if m["accuracy"] < 0.8 or min(labels.values()) < 0.7 or m["accuracy"] < m["untrained_accuracy"] + 0.05 or m["accuracy"] < m["shortcut_accuracy"] + 0.1:
            raise ValueError(name + " failed unchanged quality or baseline gates")
    runtime = report.get("runtime", {})
    if any(runtime.get(k) is not True for k in ("full_parser_passed", "reference_engine_passed", "repeat_bits_identical")) or runtime.get("negative_identity_controls", 0) < 3:
        raise ValueError("complete primary runtime and negative identity controls required")
    vectors = runtime.get("vectors", [])
    if not isinstance(vectors, list) or not 8 <= len(vectors) <= 100:
        raise ValueError("bounded actual fixed reference vectors required")
    for vector in vectors:
        tokens, score = vector.get("tokens"), vector.get("score_bits")
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= 512 or any(type(t) is not int or not 0 <= t < 50368 for t in tokens):
            raise ValueError("invalid reference tokens")
        if type(score) is not int or not 0 <= score < 2**32 or score & 0x7f800000 == 0x7f800000:
            raise ValueError("invalid/nonfinite reference score bits")
    return True
