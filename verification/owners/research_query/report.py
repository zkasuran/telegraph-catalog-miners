# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Literal wire controls and actual audit report assembly. No model imports."""
import json
import math
import os
import re
import struct

from source import sha

FOUNDATION = "0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6"
ENGINE = "77643d6e1176ab8744801cef93ccb844d0136ed32b8daa865d541f37d13292de"
TOKENIZER = "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456"


def formatted(rows, name):
    if name == "prose":
        return [dict(row) for row in rows]
    if name != "json":
        raise ValueError("unsupported literal response format")
    return [{**row, **{side: json.dumps({"answer": row[side]}, ensure_ascii=False, separators=(",", ":"))
                       for side in ("chosen", "rejected")}} for row in rows]


def bits_float(bits):
    if type(bits) is not int or not 0 <= bits < 1 << 32:
        raise ValueError("actual uint32 FP32 bits required")
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    if not math.isfinite(value):
        raise ValueError("nonfinite engine result")
    return value


def literal_reply(value, count, manifest):
    expected = {"foundation_hash": FOUNDATION, "tokenizer_hash": TOKENIZER,
                "adapter_hash": manifest["keccak256"].removeprefix("0x")}
    if not isinstance(value, dict) or any(value.get(key) != text for key, text in expected.items()):
        raise ValueError("literal bare-hash counterparty response mismatch")
    results = value.get("results")
    if not isinstance(results, list) or len(results) != count:
        raise ValueError("actual positional result count mismatch")
    values = []
    for row in results:
        if not isinstance(row, dict) or set(row) != {"score"}:
            raise ValueError("missing or conflicting score result")
        bits_float(row["score"])
        values.append(row["score"])
    return values


def accuracy(rows, scores):
    if len(rows) != 64 or len(scores) != 128 or len({r["group_id"] for r in rows}) != 32:
        raise ValueError("complete fixed independent audit required")
    if any(not math.isfinite(value) for value in scores):
        raise ValueError("nonfinite actual audit score")
    wins = [scores[2 * i] > scores[2 * i + 1] for i in range(64)]
    labels = {"Yes": [], "No": []}
    for row, won in zip(rows, wins):
        labels["Yes" if row["variant"] == "original_claim" else "No"].append(won)
    if any(len(values) != 32 for values in labels.values()):
        raise ValueError("both source labels must have 32 actual audit pairs")
    return {"accuracy": sum(wins) / 64,
            "per_label": {label: sum(values) / len(values) for label, values in labels.items()}}


def validate_ready(report):
    if report.get("schema_version") != 1 or report.get("status") != "REGISTRATION_READY":
        raise ValueError("completed registration-ready report required")
    if report.get("family") != "RESEARCH_QUERY" or type(report.get("run_id")) is not int:
        raise ValueError("exact family and actual remote run required")
    if report["run_id"] < 1 or not re.fullmatch(r"[a-f0-9]{40}", report.get("commit", "")) or report.get("serving_commit") != report["commit"]:
        raise ValueError("actual run and identical executor source commit required")
    for key in ("package_sha256", "package_keccak"):
        if not re.fullmatch(r"[a-f0-9]{64}", report.get(key, "")):
            raise ValueError("literal package hash shape invalid")
    if report.get("foundation_sha256") != FOUNDATION or report.get("engine_sha256") != ENGINE or report.get("tokenizer_sha256") != TOKENIZER:
        raise ValueError("actual pinned runtime identities required")
    if len(json.dumps(report).encode()) > 128 << 10:
        raise ValueError("bounded public registration report required")
    source = report["source"]
    if source["normalized_overlap"] != 0 or source["audit_used_for_selection"] is not False or source["grants_verified"] is not True or source["independent_groups"] < 32:
        raise ValueError("source grant or independent audit boundary failed")
    if not re.fullmatch(r"[a-f0-9]{64}", source.get("source_manifest_sha256", "")):
        raise ValueError("source manifest binding required")
    quality = report["quality"]
    if quality["passed"] is not True or quality["audit_pairs"] != 64 or quality["all_supported_formats_tested"] is not True or set(quality["formats"]) != {"prose", "json"}:
        raise ValueError("both literal supported formats must pass")
    for metrics in quality["formats"].values():
        if set(metrics["per_label"]) != {"Yes", "No"}:
            raise ValueError("both supported source labels required")
        values = [metrics["accuracy"], metrics["untrained_accuracy"], metrics["shortcut_accuracy"], *metrics["per_label"].values()]
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("finite measured format metrics required")
        if metrics["accuracy"] < .8 or min(metrics["per_label"].values()) < .7 or metrics["accuracy"] + 1e-9 < metrics["untrained_accuracy"] + .05 or metrics["accuracy"] + 1e-9 < metrics["shortcut_accuracy"] + .10:
            raise ValueError("unchanged accuracy, per-label or baseline gate failed")
    runtime = report["runtime"]
    if any(runtime.get(key) is not True for key in ("full_parser_passed", "reference_engine_passed", "repeat_bits_identical")) or runtime["negative_identity_controls"] < 3:
        raise ValueError("real parser/engine/identity execution required")
    if not 8 <= len(runtime["vectors"]) <= 100:
        raise ValueError("8-100 actual fixed vectors required")
    for vector in runtime["vectors"]:
        if not 1 <= len(vector["tokens"]) <= 256 or any(type(t) is not int or not 0 <= t < 50368 for t in vector["tokens"]):
            raise ValueError("invalid actual reference tokens")
        bits_float(vector["score_bits"])


def assemble(root, contract, manifest, runtime, untrained, shortcuts, original_result):
    if not runtime["passed"] or not original_result["quality"]["passed"]:
        raise ValueError("actual remote candidate or reference-runtime gates failed")
    package = root / "RESEARCH_QUERY.safetensors"
    observed_source = json.loads((root / "source-check.json").read_text())
    if observed_source["grants_verified"] is not True or observed_source["originals_reverified"] != 112:
        raise ValueError("actual source re-verification incomplete")
    formats = {}
    for name in ("prose", "json"):
        formats[name] = {**runtime["formats"][name], "untrained_accuracy": untrained[name]["accuracy"],
                         "shortcut_accuracy": max(shortcuts[name].values())}
    report = {"schema_version": 1, "status": "REGISTRATION_READY", "family": "RESEARCH_QUERY",
              "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
              "package_sha256": sha(package.read_bytes()), "package_keccak": manifest["keccak256"].removeprefix("0x"),
              "package_bytes": package.stat().st_size, "foundation_sha256": FOUNDATION, "engine_sha256": ENGINE,
              "tokenizer_sha256": TOKENIZER, "serving_commit": os.environ["GITHUB_SHA"],
              "source": {"normalized_overlap": observed_source["normalized_overlap"], "audit_used_for_selection": observed_source["audit_used_for_selection"], "grants_verified": observed_source["grants_verified"],
                         "source_manifest_sha256": contract["source_manifest_sha256"], "independent_groups": observed_source["independent_groups"],
                         "scope": "Complete supplied scientific excerpt agreement, not universal factual truth", "labels": "AI-assisted constructed preferences"},
              "quality": {"passed": True, "audit_pairs": 64, "all_supported_formats_tested": True, "formats": formats},
              "runtime": {"full_parser_passed": runtime["full_parser_passed"], "reference_engine_passed": runtime["passed"], "repeat_bits_identical": runtime["passed"],
                          "negative_identity_controls": runtime["negative_identity_controls"], "vectors": runtime["vectors"]}}
    validate_ready(report)
    return report
