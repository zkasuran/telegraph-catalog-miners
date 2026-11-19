# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Closed readiness gate mirroring the installed adapter without any credentials."""
import math
import re
from protocol import bare_hash, decode_bits


def validate_ready(report):
    if report.get("schema_version") != 1 or report.get("status") != "REGISTRATION_READY" or report.get("family") != "WEATHER_FORECAST_VERIFY":
        raise ValueError("missing real ready report")
    if type(report.get("run_id")) is not int or report["run_id"] < 1 or not re.fullmatch(r"[a-f0-9]{40}", report.get("commit", "")):
        raise ValueError("real run/commit identity required")
    for name in ("package_sha256", "package_keccak", "foundation_sha256", "engine_sha256", "tokenizer_sha256"):
        bare_hash(report.get(name))
    from reference import FOUNDATION
    for field, name in (("foundation_sha256", "foundation_lora_fp32.onnx"), ("engine_sha256", "foundation_engine.wasm"), ("tokenizer_sha256", "tokenizer.json")):
        if report[field] != FOUNDATION[name]:
            raise ValueError("actual pinned runtime identity differs")
    source, quality, runtime = (report.get(name, {}) for name in ("source", "quality", "runtime"))
    if source.get("normalized_overlap") != 0 or source.get("audit_used_for_selection") is not False or source.get("grants_verified") is not True or source.get("independent_groups", 0) < 32:
        raise ValueError("source/group/audit boundary incomplete")
    bare_hash(source.get("source_manifest_sha256"))
    if quality.get("passed") is not True or quality.get("audit_pairs", 0) < 64 or quality.get("all_supported_formats_tested") is not True or set(quality.get("formats", {})) != {"json", "celsius", "bare"}:
        raise ValueError("quality/format audit incomplete")
    for metrics in quality["formats"].values():
        labels = metrics.get("per_label", {})
        values = [metrics.get("accuracy"), metrics.get("untrained_accuracy"), metrics.get("shortcut_accuracy"), *labels.values()]
        if not labels or not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError("finite measured quality values required")
        accuracy = metrics["accuracy"]
        if accuracy < 0.80 or min(labels.values()) < 0.70 or accuracy + 1e-9 < metrics["untrained_accuracy"] + 0.05 or accuracy + 1e-9 < metrics["shortcut_accuracy"] + 0.10:
            raise ValueError("unchanged accuracy and improvement gates failed")
    if any(runtime.get(name) is not True for name in ("full_parser_passed", "reference_engine_passed", "repeat_bits_identical")) or runtime.get("negative_identity_controls", 0) < 3:
        raise ValueError("actual reference-runtime/identity proof incomplete")
    vectors = runtime.get("vectors", [])
    if not isinstance(vectors, list) or not 8 <= len(vectors) <= 100:
        raise ValueError("bounded actual fixed vectors required")
    for vector in vectors:
        tokens = vector.get("tokens")
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= 512 or not all(type(t) is int and 0 <= t < 50368 for t in tokens):
            raise ValueError("invalid real reference vector")
        decode_bits(vector.get("score_bits"))
    if report.get("serving_commit") != report["commit"]:
        raise ValueError("identical source executor commit required")
    return True
