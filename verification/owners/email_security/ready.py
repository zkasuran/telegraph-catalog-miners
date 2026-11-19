# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Registration-ready contract assembled only from completed actual results."""
import math
import re

from runtime import HASHES

SHA = re.compile(r"[a-f0-9]{64}")


def finite_accuracy(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def quality_gate(accuracy, labels, untrained, shortcut):
    failures = []
    if not labels or not all(finite_accuracy(v) for v in (accuracy, untrained, shortcut, *labels.values())):
        return ["invalid or missing completed quality metric"]
    if accuracy < .80 or min(labels.values()) < .70:
        failures.append("80 percent overall or 70 percent per-label gate")
    if accuracy + 1e-9 < untrained + .05:
        failures.append("five points above seed-matched untrained gate")
    if accuracy + 1e-9 < shortcut + .10:
        failures.append("ten points above strongest shortcut gate")
    return failures


def verify_ready(report):
    if not isinstance(report, dict) or report.get("schema_version") != 1 or report.get("status") != "REGISTRATION_READY":
        raise ValueError("completed registration-ready report required")
    if report.get("family") != "EMAIL_SECURITY" or type(report.get("run_id")) is not int or report["run_id"] < 1:
        raise ValueError("actual family and GitHub run identity required")
    if not re.fullmatch(r"[a-f0-9]{40}", report.get("commit", "")) or report.get("serving_commit") != report["commit"]:
        raise ValueError("source commit and identical public serving commit required")
    for key in ("package_sha256", "package_keccak"):
        if not SHA.fullmatch(report.get(key, "")):
            raise ValueError("full bare candidate hash required")
    for key, name in (("foundation_sha256", "foundation_lora_fp32.onnx"), ("engine_sha256", "foundation_engine.wasm"),
                      ("tokenizer_sha256", "tokenizer.json")):
        if report.get(key) != HASHES[name]:
            raise ValueError("pinned runtime identity mismatch")
    source = report.get("source", {})
    if (source.get("normalized_overlap") != 0 or source.get("audit_used_for_selection") is not False
            or source.get("grants_verified") is not True or not SHA.fullmatch(source.get("source_manifest_sha256", ""))
            or source.get("independent_groups", 0) < 32):
        raise ValueError("source grant and grouped audit boundary required")
    quality = report.get("quality", {})
    if quality.get("passed") is not True or quality.get("audit_pairs", 0) < 64 or quality.get("all_supported_formats_tested") is not True:
        raise ValueError("completed quality and all-format audit required")
    if set(quality.get("formats", {})) != {"dns_json"}:
        raise ValueError("only actual raw DNS JSON format is claimed")
    for metrics in quality["formats"].values():
        if quality_gate(metrics.get("accuracy"), metrics.get("per_label", {}), metrics.get("untrained_accuracy"), metrics.get("shortcut_accuracy")):
            raise ValueError("completed format failed quality gate")
    runtime = report.get("runtime", {})
    if any(runtime.get(k) is not True for k in ("full_parser_passed", "reference_engine_passed", "repeat_bits_identical")):
        raise ValueError("actual full parser and reference execution required")
    if runtime.get("negative_identity_controls", 0) < 3 or not 8 <= len(runtime.get("vectors", [])) <= 100:
        raise ValueError("actual identity controls and 8-100 fixed engine vectors required")
    for vector in runtime["vectors"]:
        bits, tokens = vector.get("score_bits"), vector.get("tokens")
        if (type(bits) is not int or not 0 <= bits < 1 << 32 or bits & 0x7f800000 == 0x7f800000
                or not isinstance(tokens, list) or not 1 <= len(tokens) <= 256
                or any(type(token) is not int or not 0 <= token < 50368 for token in tokens)):
            raise ValueError("finite real engine bits and token vectors required")
    return report


def build_ready(identity, source, manifest, runtime, untrained, controls, answer_only, hashes):
    if runtime.get("passed") is not True or manifest.get("manifest", {}).get("verification") != {"stage1": True, "loaded": False}:
        raise ValueError("actual full parser or engine has not passed")
    shortcut = max(controls["length"], controls["answer_only"], controls["literal_word_overlap"],
                   controls["token_length"], answer_only["pairwise_accuracy"])
    accuracy, labels = runtime["pairwise_accuracy"], runtime["per_label"]
    failures = quality_gate(accuracy, labels, untrained["pairwise_accuracy"], shortcut)
    if failures:
        raise ValueError("; ".join(failures))
    report = {"SPDX-License-Identifier": "LicenseRef-zkasuran-SAND-1.0",
              "schema_version": 1, "status": "REGISTRATION_READY", "family": "EMAIL_SECURITY",
              "run_id": int(identity["github_run_id"]), "commit": identity["commit"], "serving_commit": identity["commit"],
              "package_sha256": manifest["sha256"], "package_keccak": manifest["keccak256"].removeprefix("0x"),
              "package_bytes": manifest["bytes"], "foundation_sha256": HASHES["foundation_lora_fp32.onnx"],
              "engine_sha256": HASHES["foundation_engine.wasm"], "tokenizer_sha256": HASHES["tokenizer.json"],
              "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True,
                         "source_manifest_sha256": hashes["source"], "independent_groups": source["files"]["audit"]["groups"],
                         "kind": "AI-assisted constructed protocol fixtures", "external_dns_truth": False,
                         "scope": "raw DNS Question identity and internal response consistency only"},
              "quality": {"passed": True, "audit_pairs": source["files"]["audit"]["rows"], "all_supported_formats_tested": True,
                          "formats": {"dns_json": {"accuracy": accuracy, "per_label": labels,
                                      "untrained_accuracy": untrained["pairwise_accuracy"], "shortcut_accuracy": shortcut}},
                          "exact_parser_reference_accuracy": controls["exact_question_parser"], "controls": controls},
              "runtime": {"full_parser_passed": True, "reference_engine_passed": True,
                          "repeat_bits_identical": runtime["repeat_bits_identical"],
                          "negative_identity_controls": runtime["wrong_identity_controls"], "vectors": runtime["vectors"]},
              "proof_sha256": hashes, "limits": ["Self-reported DNS is not authenticated external DNS truth.",
                         "No human review or independent evaluation contributor is claimed.",
                         "Training input is the literal documented request and raw DNS JSON. Actual node quality remains a fresh qualification gate."]}
    return verify_ready(report)
