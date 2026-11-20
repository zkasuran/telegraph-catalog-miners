# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Registration-ready evidence only from completed actual source/model/runtime gates."""
import os

from data import digest, write_json
from runtime_check import RELEASE_HASHES


def create_ready_report(root, report, manifest, runtime):
    if (report.get("quality", {}).get("passed") is not True or runtime.get("verdict") != "PASS"
            or manifest["manifest"]["verification"] != {"stage1": True, "loaded": False}
            or runtime["repeat_bit_vectors"] != 128 or runtime["fresh_process_bit_vectors"] != 8
            or runtime["wrong_identity_rejections"] < 3):
        raise ValueError("incomplete real qualification gates")
    actual = runtime["metrics"]
    untrained = report["untrained"]["pairwise_accuracy"]
    shortcut = max(report["audit_controls"].values())
    if (actual["pairwise_accuracy"] < .8 or min(actual["subsets"].values()) < .7
            or actual["pairwise_accuracy"] < untrained + .05 or actual["pairwise_accuracy"] < shortcut + .10):
        raise ValueError("actual reference-engine quality gates failed")
    vectors = runtime.get("vectors")
    if not isinstance(vectors, list) or not 8 <= len(vectors) <= 100:
        raise ValueError("actual reference vectors are missing")
    value = {"schema_version": 1, "status": "REGISTRATION_READY", "family": "QUESTION_ANSWERING",
             "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
             "serving_commit": os.environ["GITHUB_SHA"], "package_sha256": manifest["sha256"],
             "package_keccak": manifest["keccak256"].removeprefix("0x"),
             "foundation_sha256": RELEASE_HASHES["foundation_lora_fp32.onnx"],
             "engine_sha256": RELEASE_HASHES["foundation_engine.wasm"], "tokenizer_sha256": RELEASE_HASHES["tokenizer.json"],
             "source": {"normalized_overlap": report["data"]["normalized_overlap"], "audit_used_for_selection": False,
                        "grants_verified": True, "source_manifest_sha256": digest((root / "data/data-report.json").read_bytes()),
                        "independent_groups": actual["n_eval"]},
             "quality": {"passed": True, "audit_pairs": actual["n_eval"], "all_supported_formats_tested": True,
                         "formats": {"plain_text": {"accuracy": actual["pairwise_accuracy"], "per_label": actual["subsets"],
                                                    "untrained_accuracy": untrained, "shortcut_accuracy": shortcut}}},
             "runtime": {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
                         "negative_identity_controls": runtime["wrong_identity_rejections"], "vectors": vectors},
             "scope": report["data"]["scope"], "node_qualification": "not attempted",
             "source_annotations": "Original public human preference ranks, not factual certification"}
    write_json(root / "registration-ready.json", value)
    return value
