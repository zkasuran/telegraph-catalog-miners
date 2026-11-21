# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Registration readiness requires observed source, quality and engine evidence."""
import importlib.metadata
import math
import os
import re
import socket
import sys

from data import FAMILY, VIEWS, write_json
from metrics import all_formats
from runtime_check import RELEASE_HASHES, model_remote_only
from training_policy import EPOCHS
from wire import decode_bits

SHA = re.compile(r"[a-f0-9]{64}")
COMMIT = re.compile(r"[a-f0-9]{40}")


def validate_ready(value):
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int or value.get("schema_version") != 1 or value.get("status") != "REGISTRATION_READY"
            or value.get("family") != FAMILY or type(value.get("run_id")) is not int or value["run_id"] < 1
            or not COMMIT.fullmatch(value.get("commit", "")) or value.get("serving_commit") != value["commit"]):
        raise ValueError("ready run, family or serving identity mismatch")
    for name in ("package_sha256", "package_keccak"):
        if not SHA.fullmatch(value.get(name, "")):
            raise ValueError("literal bare package hash required")
    for field, name in (("foundation_sha256", "foundation_lora_fp32.onnx"), ("engine_sha256", "foundation_engine.wasm"), ("tokenizer_sha256", "tokenizer.json")):
        if value.get(field) != RELEASE_HASHES[name]:
            raise ValueError("ready pinned runtime identity mismatch")
    source = value.get("source", {})
    if (type(source.get("normalized_overlap")) is not int or source.get("normalized_overlap") != 0 or source.get("audit_used_for_selection") is not False
            or source.get("grants_verified") is not True or type(source.get("independent_groups")) is not int or source.get("independent_groups") != 64
            or not SHA.fullmatch(source.get("source_manifest_sha256", ""))):
        raise ValueError("ready source and independent audit boundary failed")
    quality = value.get("quality", {})
    if (quality.get("passed") is not True or type(quality.get("audit_pairs")) is not int or quality.get("audit_pairs") != 128 or quality.get("all_supported_formats_tested") is not True
            or set(quality.get("formats", {})) != set(VIEWS)):
        raise ValueError("ready complete claimed format audit required")
    for metrics in quality["formats"].values():
        labels = metrics.get("per_label", {})
        values = [metrics.get("accuracy"), metrics.get("untrained_accuracy"), metrics.get("shortcut_accuracy"), *labels.values()]
        if set(labels) != {"A", "B"} or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("ready finite target-label metrics required")
        if (metrics["accuracy"] < .8 or min(labels.values()) < .7 or metrics["shortcut_accuracy"] > .65
                or metrics["accuracy"] + 1e-9 < metrics["untrained_accuracy"] + .05
                or metrics["accuracy"] + 1e-9 < metrics["shortcut_accuracy"] + .10):
            raise ValueError("ready unchanged quality thresholds failed")
    runtime = value.get("runtime", {})
    if (any(runtime.get(k) is not True for k in ("full_parser_passed", "reference_engine_passed", "repeat_bits_identical"))
            or type(runtime.get("negative_identity_controls")) is not int or runtime.get("negative_identity_controls", 0) < 3
            or not isinstance(runtime.get("vectors"), list) or not 8 <= len(runtime.get("vectors", [])) <= 100):
        raise ValueError("ready actual parser, engine and repeated vectors required")
    for vector in runtime["vectors"]:
        tokens = vector.get("tokens")
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= 256 or any(type(t) is not int or not 0 <= t < 50368 for t in tokens):
            raise ValueError("ready bounded reference vector required")
        decode_bits(vector.get("score_bits"))
    return value


def completed_checks(report, primary):
    runtime, package = report["runtime"], report["package_manifest"]
    quality = all_formats(runtime["metrics"], report["untrained"], report["audit_controls"])
    if (report["verdict"] != "PASS" or not quality["passed"] or report["epochs_completed"] != EPOCHS
            or report["hyperparameters"]["epochs"] != EPOCHS or len(report["development_history"]) != EPOCHS
            or report["selection_split"] != "validation" or not report["dataset_gate"]["ok"]
            or not primary["grants_verified"] or primary["primary_documents"] != 448
            or primary["source_manifest_sha256"] != report["data"]["source_manifest_sha256"]
            or package["manifest"]["verification"] != {"stage1": True, "loaded": False}):
        raise ValueError("incomplete source, selected model or reference runtime cannot become ready")
    return quality


def registration_ready(root, report, primary, training_origin=None):
    model_remote_only()
    quality = completed_checks(report, primary)
    runtime, package = report["runtime"], report["package_manifest"]
    value = {"schema_version": 1, "status": "REGISTRATION_READY", "family": FAMILY,
             "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
             "serving_commit": os.environ["GITHUB_SHA"], "package_sha256": package["sha256"],
             "package_keccak": package["keccak256"].removeprefix("0x"),
             "foundation_sha256": RELEASE_HASHES["foundation_lora_fp32.onnx"],
             "engine_sha256": RELEASE_HASHES["foundation_engine.wasm"], "tokenizer_sha256": RELEASE_HASHES["tokenizer.json"],
             "source": {"normalized_overlap": report["data"]["boundary"]["normalized_overlap"],
                        "audit_used_for_selection": False, "grants_verified": primary["grants_verified"],
                        "source_manifest_sha256": primary["source_manifest_sha256"],
                        "independent_groups": report["data"]["boundary"]["audit_groups"]},
             "quality": {"passed": quality["passed"], "audit_pairs": runtime["metrics"]["raw"]["n_eval"],
                         "all_supported_formats_tested": set(runtime["metrics"]) == set(VIEWS),
                         "formats": {style: {"accuracy": runtime["metrics"][style]["pairwise_accuracy"],
                                             "per_label": runtime["metrics"][style]["per_label"],
                                             "untrained_accuracy": report["untrained"][style]["pairwise_accuracy"],
                                             "shortcut_accuracy": max(report["audit_controls"][style].values())} for style in VIEWS}},
             "runtime": {"full_parser_passed": package["manifest"]["verification"]["stage1"],
                         "reference_engine_passed": runtime["verdict"] == "PASS", "repeat_bits_identical": runtime["repeat_bits_identical"],
                         "negative_identity_controls": runtime["wrong_identity_rejections"], "vectors": runtime["vectors"]},
             "scope": report["data"]["scope"], "baseline_runtime": "seed-matched native CPU initialization",
             "node": "https://devnode.telegraphprotocol.com", "runtime_node": socket.gethostname(),
             "python": sys.version, "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "transformers", "peft", "tokenizers")},
             "owner_attempt": os.environ["OWNER_ATTEMPT"],
              "node_qualification": "not attempted"}
    if training_origin is not None:
        value["training_origin"] = training_origin
        value["audit_origin"] = {"run_id": training_origin["run_id"], "commit": training_origin["commit"],
                                 "scored_once": True, "runtime_report_sha256": training_origin["runtime_report_sha256"]}
        value["recovery_scope"] = "Original immutable quality evidence plus fresh full parser and fixed-vector reference-engine verification. No fitting or audit rescoring."
    validate_ready(value)
    write_json(root / "registration-ready.json", value)
    return value
