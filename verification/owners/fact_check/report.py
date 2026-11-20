# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Registration report from completed measured results only, never expectations."""
import importlib.metadata
import json
import math
import os
import platform
import re

from data import HERE, digest, read_splits, write_json
from licensing import licence_metadata
from runtime_check import RELEASE_HASHES, decode_bits, model_remote_only


def require_metrics(metrics):
    accuracy = metrics.get("accuracy")
    labels = metrics.get("per_label", {})
    values = [accuracy, metrics.get("untrained_accuracy"), metrics.get("shortcut_accuracy"), *labels.values()]
    if set(labels) != {"True", "False"} or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("measured format/label metrics missing or invalid")
    if (accuracy < .8 or min(labels.values()) < .7 or accuracy + 1e-9 < metrics["untrained_accuracy"] + .05
            or accuracy + 1e-9 < metrics["shortcut_accuracy"] + .10 or metrics["shortcut_accuracy"] > .65):
        raise ValueError("unchanged measured quality gates failed")


def validate_ready(report):
    if report.get("schema_version") != 1 or report.get("status") != "REGISTRATION_READY" or report.get("family") != "FACT_CHECK":
        raise ValueError("completed registration report required")
    if type(report.get("run_id")) is not int or report["run_id"] < 1:
        raise ValueError("actual dispatched run id required")
    if not re.fullmatch(r"[a-f0-9]{40}", report.get("commit", "")):
        raise ValueError("actual immutable source commit required")
    for name in ("package_sha256", "package_keccak", "foundation_sha256", "engine_sha256", "tokenizer_sha256"):
        value = report.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("literal bare lowercase hash required")
    for key, file in (("foundation_sha256", "foundation_lora_fp32.onnx"), ("engine_sha256", "foundation_engine.wasm"), ("tokenizer_sha256", "tokenizer.json")):
        if report[key] != RELEASE_HASHES[file]:
            raise ValueError("foundation identity changed")
    source = report.get("source", {})
    if (source.get("normalized_overlap") != 0 or source.get("audit_used_for_selection") is not False
            or source.get("grants_verified") is not True or source.get("independent_groups", 0) < 64
            or not re.fullmatch(r"[a-f0-9]{64}", source.get("source_manifest_sha256", ""))):
        raise ValueError("source and frozen independent audit proof missing")
    quality = report.get("quality", {})
    if quality.get("passed") is not True or quality.get("audit_pairs") != 64 or quality.get("all_supported_formats_tested") is not True or set(quality.get("formats", {})) != {"prose"}:
        raise ValueError("complete supported-format audit required")
    require_metrics(quality["formats"]["prose"])
    runtime = report.get("runtime", {})
    if any(runtime.get(key) is not True for key in ("full_parser_passed", "reference_engine_passed", "repeat_bits_identical")) or runtime.get("negative_identity_controls", 0) < 3:
        raise ValueError("actual parser/reference/repetition proof required")
    vectors = runtime.get("vectors", [])
    if not 8 <= len(vectors) <= 100:
        raise ValueError("actual fixed vectors missing")
    for vector in vectors:
        tokens = vector.get("tokens")
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= 256 or any(type(t) is not int or not 0 <= t < 50368 for t in tokens):
            raise ValueError("invalid reference vector tokens")
        decode_bits(vector.get("score_bits"))
    if report.get("serving_commit") != report.get("commit"):
        raise ValueError("identical serving source commit required")
    return report


def emit(root):
    model_remote_only()
    result = json.loads((root / "result.json").read_text())
    runtime = json.loads((root / "runtime-result.json").read_text())
    package = json.loads((root / "package-manifest.json").read_text())
    data, splits = read_splits(root / "data")
    if data["source_manifest_sha256"] != digest((HERE / "sources.json").read_bytes()):
        raise ValueError("actual source manifest differs from the frozen source gate")
    vectors = json.loads((root / "runtime-vectors.json").read_text())["vectors"]
    streams = json.loads((root / "runtime-score-bits.json").read_text())["streams"]
    if (result.get("verdict") != "PASS" or runtime.get("verdict") != "PASS" or result.get("quality", {}).get("passed") is not True
            or result.get("epochs_completed") != 3 or result.get("selection_split") != "validation"
            or len(streams) != 2 or streams[0] != streams[1] or len(streams[0]) != 128
            or runtime.get("fresh_process_bit_vectors") != 8):
        raise ValueError("measured completed three-epoch audit/runtime required")
    rows = splits["audit"]
    wins = [decode_bits(streams[0][2*i]) > decode_bits(streams[0][2*i+1]) for i in range(len(rows))]
    per_label = {label: sum(win for row, win in zip(rows, wins) if row["label"] == label) / sum(row["label"] == label for row in rows)
                 for label in ("True", "False")}
    path = root / "FACT_CHECK.safetensors"
    if digest(path.read_bytes()) != package["sha256"] or runtime["package_sha256"] != package["sha256"]:
        raise ValueError("audited package bytes differ")
    node = json.loads((root / "class-capture.json").read_text())
    report = {
        "schema_version": 1, "status": "REGISTRATION_READY", "family": "FACT_CHECK",
        "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
        "serving_commit": os.environ["GITHUB_SHA"], "package_sha256": package["sha256"],
        "package_keccak": package["keccak256"].removeprefix("0x"), "package_bytes": path.stat().st_size,
        "foundation_sha256": RELEASE_HASHES["foundation_lora_fp32.onnx"],
        "engine_sha256": RELEASE_HASHES["foundation_engine.wasm"], "tokenizer_sha256": RELEASE_HASHES["tokenizer.json"],
        "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": data["training_allowed"],
                   "source_manifest_sha256": digest((HERE / "sources.json").read_bytes()), "independent_groups": len(rows),
                   "scope": data["scope"], "audit_sha256": data["files"]["audit"]["sha256"],
                   "preference_authorship": "AI-assisted deterministic source/claim verdict construction, not human preference labels"},
        "quality": {"passed": True, "audit_pairs": len(rows), "all_supported_formats_tested": True,
                    "formats": {"prose": {"accuracy": sum(wins)/len(wins), "per_label": per_label,
                                          "untrained_accuracy": result["untrained"]["pairwise_accuracy"],
                                          "shortcut_accuracy": max(result["audit_controls"].values())}}},
        "runtime": {"full_parser_passed": package["manifest"]["verification"]["stage1"],
                    "reference_engine_passed": runtime["verdict"] == "PASS", "repeat_bits_identical": streams[0] == streams[1],
                    "negative_identity_controls": runtime["wrong_identity_rejections"], "vectors": vectors},
        "node_identity": node, "runtime_identity": {"python": platform.python_version(), "platform": platform.platform(),
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy", "tokenizers", "safetensors")}},
        "licences": licence_metadata(), "node_qualification": "not attempted",
    }
    validate_ready(report)
    write_json(root / "registration-ready.json", report)
    if (root / "registration-ready.json").stat().st_size > 256 << 10:
        raise ValueError("registration report byte ceiling")
    return report
