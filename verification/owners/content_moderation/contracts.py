# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Closed source/quality/wire contracts. Standard library only."""
import hashlib
import json
import math
import os
import re
import struct
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = "CONTENT_MODERATION"
CHILD = FAMILY + "_GENERAL"
REPO = "zkasuran/telegraph-catalog-miners"
WORKFLOW = "evaluator-owner-content_moderation.yml"
FOUNDATION = {"foundation_lora_fp32.onnx": "0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6",
              "foundation_engine.wasm": "77643d6e1176ab8744801cef93ccb844d0136ed32b8daa865d541f37d13292de",
              "tokenizer.json": "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456"}
EXPERIMENT = json.loads((HERE / "experiment.json").read_text())
FORMATS = tuple(EXPERIMENT["formats"])
LABELS = set(EXPERIMENT["labels"])
SHA = re.compile(r"[0-9a-f]{64}")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def strict_json(raw):
    def fields(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON constant")
    return json.loads(raw, object_pairs_hook=fields, parse_constant=invalid)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(canonical(value) + b"\n")
    temporary.replace(path)


def remote_only():
    required = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO,
                "RUNNER_ENVIRONMENT": "github-hosted", "RUNNER_OS": "Linux",
                "GITHUB_EVENT_NAME": "workflow_dispatch", "DIAGNOSTIC_PUBLIC_REPOSITORY": "true"}
    reference = os.environ.get("GITHUB_WORKFLOW_REF", "")
    if any(os.environ.get(k) != v for k, v in required.items()) or not reference.startswith(REPO + "/.github/workflows/" + WORKFLOW + "@"):
        raise RuntimeError("model work requires the owned free public CPU workflow")


def bits(value):
    if type(value) is not int or not 0 <= value < 1 << 32:
        raise ValueError("invalid FP32 score bits")
    result = struct.unpack("<f", struct.pack("<I", value))[0]
    if not math.isfinite(result):
        raise ValueError("nonfinite FP32 score")
    return result


def identity(package):
    if not isinstance(package, str) or not SHA.fullmatch(package):
        raise ValueError("bare lowercase package keccak required")
    return {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
            "adapter_hash": package, "tokenizer_hash": FOUNDATION["tokenizer.json"]}


def input_root(batch):
    # Literal upstream remote.BatchInputRoot expression, including sorted indices.
    leaves = [str(i) + ":" + sha(json.dumps({"input_ids": row["tokens"], "attention_mask": [1] * len(row["tokens"])}, separators=(",", ":")).encode()) for i, row in enumerate(batch)]
    return sha("\n".join(sorted(leaves)).encode()) if leaves else ""


def check_reply(value, request, package):
    expected = identity(package)
    if not isinstance(value, dict) or any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("literal bare-hash wire echo mismatch")
    if request.get("input_root") and value.get("input_root") != request["input_root"]:
        raise ValueError("wire input-root mismatch")
    scores = value.get("results")
    if not isinstance(scores, list) or len(scores) != len(request["batch"]):
        raise ValueError("wire result count mismatch")
    result = []
    for row in scores:
        if not isinstance(row, dict) or set(row) != {"score"}:
            raise ValueError("missing or conflicting wire score")
        bits(row["score"])
        result.append(row["score"])
    return result


def eligibility(formats, untrained, shortcuts):
    failures = []
    if set(formats) != set(FORMATS) or set(untrained) != set(FORMATS) or set(shortcuts) != set(FORMATS):
        return ["missing claimed format or baseline"]
    for name in FORMATS:
        row = formats[name]
        if set(row.get("by_label", {})) != LABELS:
            failures.append(name + " missing label")
            continue
        values = [row["accuracy"], untrained[name]["accuracy"], shortcuts[name], *[v["accuracy"] for v in row["by_label"].values()]]
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            failures.append(name + " invalid metric")
            continue
        if row["accuracy"] < 0.8 or min(v["accuracy"] for v in row["by_label"].values()) < 0.7:
            failures.append(name + " below overall/per-label gate")
        if row["accuracy"] + 1e-12 < untrained[name]["accuracy"] + 0.05:
            failures.append(name + " below untrained improvement gate")
        if row["accuracy"] + 1e-12 < shortcuts[name] + 0.10:
            failures.append(name + " below strongest shortcut gate")
    return failures


def validate_experiment(value):
    if value.get("family") != FAMILY or value.get("child") != CHILD or value.get("class") != "judgment":
        raise ValueError("owned family/class mismatch")
    if value.get("epochs") != 9 or value.get("audit_originals") != 256 or value.get("job_minutes") != 330:
        raise ValueError("frozen schedule/audit/budget mismatch")
    if set(value.get("formats", [])) != set(FORMATS) or set(value.get("labels", [])) != LABELS:
        raise ValueError("missing format/label coverage")
    if value.get("historical_comparison_valid") is not False:
        raise ValueError("historical failure relabelled")
    expected = {"seed": 42, "lora_rank": 8, "lora_alpha": 32, "lr": 0.0001, "lora_dropout": 0.1,
                "weight_decay": 0.01, "batch_size": 2, "grad_accum": 8, "max_length": 256,
                "early_stopping": True, "es_min_delta": 0.001, "es_patience": 1}
    if value.get("hyperparameters") != expected:
        raise ValueError("frozen rank/LR/dropout/source training recipe changed")
    for key in ("audit_sha256", "exclusion_sha256"):
        if not SHA.fullmatch(value.get(key, "")):
            raise ValueError("unbound sealed source boundary")


def check_class(snapshot):
    if set(snapshot) != {"schema", "fetched_at", "family", "child"} or snapshot["schema"] != 1:
        raise ValueError("class snapshot schema mismatch")
    moment = datetime.fromisoformat(snapshot["fetched_at"].replace("Z", "+00:00"))
    if moment.tzinfo is None or not -60 <= (datetime.now(timezone.utc) - moment).total_seconds() <= 900:
        raise ValueError("class snapshot stale or unzoned")
    for name, url in (("family", "https://devnode.telegraphprotocol.com/api/wasm?resolve=" + FAMILY),
                      ("child", "https://devnode.telegraphprotocol.com/engine/v1/intents/" + CHILD)):
        if set(snapshot[name]) != {"url", "raw"} or snapshot[name]["url"] != url:
            raise ValueError("authoritative class URL mismatch")
    parent = strict_json(snapshot["family"]["raw"])["resolve"]
    child = strict_json(snapshot["child"]["raw"])
    if parent.get("intent") != FAMILY or parent.get("class") != "judgment" or child.get("intent_id") != CHILD or child.get("intent_name") != CHILD or child.get("class") != "judgment" or child.get("evaluation", {}).get("kind") != "judgment":
        raise ValueError("live family/child/class mismatch")
    if any(child.get(name, {}).get("served_by") != FAMILY for name in ("request", "response", "onchain")) or child.get("onchain", {}).get("ready") is not True:
        raise ValueError("live child serving family mismatch")


def validate_report(report):
    if report.get("schema_version") != 1 or report.get("status") != "REGISTRATION_READY" or report.get("family") != FAMILY:
        raise ValueError("actual passing owner report required")
    if type(report.get("run_id")) is not int or report["run_id"] < 1 or not re.fullmatch(r"[0-9a-f]{40}", report.get("commit", "")) or report.get("serving_commit") != report["commit"]:
        raise ValueError("actual run and identical source commit required")
    for key in ("package_sha256", "package_keccak"):
        if not SHA.fullmatch(report.get(key, "")):
            raise ValueError("report package identity missing")
    if type(report.get("package_bytes")) is not int or not 8 < report["package_bytes"] <= 64 << 20:
        raise ValueError("actual bounded package bytes required")
    for key, value in (("foundation_sha256", FOUNDATION["foundation_lora_fp32.onnx"]), ("engine_sha256", FOUNDATION["foundation_engine.wasm"]), ("tokenizer_sha256", FOUNDATION["tokenizer.json"])):
        if report.get(key) != value:
            raise ValueError("report runtime identity swapped")
    source = report.get("source", {})
    if source.get("normalized_overlap") != 0 or source.get("audit_used_for_selection") is not False or source.get("grants_verified") is not True or source.get("independent_groups", 0) < 32 or not SHA.fullmatch(source.get("source_manifest_sha256", "")):
        raise ValueError("report independent source boundary failed")
    quality = report.get("quality", {})
    if quality.get("passed") is not True or quality.get("audit_pairs", 0) < 64 or quality.get("all_supported_formats_tested") is not True or set(quality.get("formats", {})) != set(FORMATS):
        raise ValueError("report complete audit missing")
    for row in quality["formats"].values():
        values = [row["accuracy"], row["untrained_accuracy"], row["shortcut_accuracy"], *row["per_label"].values()]
        if set(row["per_label"]) != LABELS or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("report label/finite metrics missing")
        if row["accuracy"] < 0.8 or min(row["per_label"].values()) < 0.7 or row["accuracy"] + 1e-12 < row["untrained_accuracy"] + 0.05 or row["accuracy"] + 1e-12 < row["shortcut_accuracy"] + 0.1:
            raise ValueError("report quality threshold failed")
    runtime = report.get("runtime", {})
    if any(runtime.get(k) is not True for k in ("full_parser_passed", "reference_engine_passed", "repeat_bits_identical")) or runtime.get("negative_identity_controls", 0) < 3:
        raise ValueError("report runtime/identity gates missing")
    vectors = runtime.get("vectors", [])
    if not 8 <= len(vectors) <= 100:
        raise ValueError("report fixed reference vector count failed")
    for vector in vectors:
        if not isinstance(vector.get("tokens"), list) or not 1 <= len(vector["tokens"]) <= 512 or any(type(t) is not int or not 0 <= t < 50368 for t in vector["tokens"]):
            raise ValueError("report invalid fixed token vector")
        bits(vector["score_bits"])
    return vectors
