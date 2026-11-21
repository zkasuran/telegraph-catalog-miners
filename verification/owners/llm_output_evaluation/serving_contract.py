# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only serving preflight for the actual registration-ready wire contract."""
import math
import re

from common import DEFAULT_OWNER, FOUNDATION
from data import FAMILY, FORMATS, RATINGS
from proxy import score_bits


HASH = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")


def valid_hash(value):
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def valid_metric(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def validate_ready(report, package_sha256, package_keccak, serving_commit):
    if (not isinstance(report, dict) or type(report.get("schema_version")) is not int or report["schema_version"] != 1
            or report.get("status") != "REGISTRATION_READY" or report.get("family") != FAMILY
            or not valid_hash(package_sha256) or not valid_hash(package_keccak)
            or not isinstance(serving_commit, str) or COMMIT.fullmatch(serving_commit) is None
            or report.get("package_sha256") != package_sha256 or report.get("package_keccak") != package_keccak
            or report.get("serving_commit") != serving_commit
            or type(report.get("run_id")) is not int or report["run_id"] < 1
            or not isinstance(report.get("commit"), str) or COMMIT.fullmatch(report["commit"]) is None):
        raise ValueError("serving registration-ready identity mismatch")
    identities = {"foundation_sha256": FOUNDATION["foundation_lora_fp32.onnx"],
                  "engine_sha256": FOUNDATION["foundation_engine.wasm"],
                  "tokenizer_sha256": FOUNDATION["tokenizer.json"]}
    if any(report.get(key) != value for key, value in identities.items()):
        raise ValueError("serving pinned runtime identity mismatch")
    owner = report.get("owner", DEFAULT_OWNER)
    if not isinstance(owner, str) or owner.lower() != DEFAULT_OWNER.lower():
        raise ValueError("serving selected owner mismatch")
    source = report.get("source", {})
    if (not isinstance(source, dict) or type(source.get("normalized_overlap")) is not int
            or source["normalized_overlap"] != 0 or source.get("audit_used_for_selection") is not False
            or source.get("grants_verified") is not True or not valid_hash(source.get("source_manifest_sha256"))
            or type(source.get("independent_groups")) is not int or source["independent_groups"] < 32):
        raise ValueError("serving independent source proof required")
    quality = report.get("quality", {})
    if (not isinstance(quality, dict) or quality.get("passed") is not True
            or type(quality.get("audit_pairs")) is not int or quality["audit_pairs"] < 64
            or quality.get("all_supported_formats_tested") is not True
            or not isinstance(quality.get("formats"), dict) or set(quality["formats"]) != set(FORMATS)):
        raise ValueError("serving complete frozen format audit required")
    for metrics in quality["formats"].values():
        if not isinstance(metrics, dict) or not isinstance(metrics.get("per_label"), dict):
            raise ValueError("serving finite audit metrics required")
        labels = metrics["per_label"]
        values = [metrics.get("accuracy"), metrics.get("untrained_accuracy"), metrics.get("shortcut_accuracy"), *labels.values()]
        if set(labels) != set(RATINGS) or not all(valid_metric(value) for value in values):
            raise ValueError("serving all rating labels and finite audit metrics required")
        if (metrics["accuracy"] < 0.80 or min(labels.values()) < 0.70
                or metrics["accuracy"] + 1e-9 < metrics["untrained_accuracy"] + 0.05
                or metrics["accuracy"] + 1e-9 < metrics["shortcut_accuracy"] + 0.10
                or metrics["shortcut_accuracy"] > 0.65):
            raise ValueError("serving unchanged quality gates failed")
    runtime = report.get("runtime", {})
    if (not isinstance(runtime, dict) or runtime.get("full_parser_passed") is not True
            or runtime.get("reference_engine_passed") is not True or runtime.get("repeat_bits_identical") is not True
            or type(runtime.get("negative_identity_controls")) is not int or runtime["negative_identity_controls"] < 3):
        raise ValueError("serving full reference runtime proof required")
    vectors = runtime.get("vectors")
    if not isinstance(vectors, list) or not 8 <= len(vectors) <= 100:
        raise ValueError("serving bounded actual reference vectors required")
    for vector in vectors:
        if not isinstance(vector, dict) or set(vector) != {"tokens", "score_bits"}:
            raise ValueError("serving reference vector schema mismatch")
        tokens = vector["tokens"]
        if (not isinstance(tokens, list) or not 1 <= len(tokens) <= 512
                or any(type(token) is not int or not 0 <= token < 50368 for token in tokens)):
            raise ValueError("serving bounded reference tokens required")
        score_bits(vector["score_bits"])
    return vectors
