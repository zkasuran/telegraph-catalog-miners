# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Strict measurement/ready gates. No neural imports or inferred successes."""
import math
import re
from collections import defaultdict

from wire import FOUNDATION, ENGINE, TOKENIZER, score_value


def metrics(rows, bits, format_name):
    if len(rows) != 64 or len(bits) != 128 or len({r["block_id"] for r in rows}) != 64:
        raise ValueError("all 64 independent audit originals required")
    values = [score_value(v) for v in bits]
    wins = [values[i] > values[i + 1] for i in range(0, 128, 2)]
    groups = defaultdict(list)
    for row, win in zip(rows, wins):
        a, b = (row["formats"][format_name][side] for side in ("chosen", "rejected"))
        label = "chosen_longer" if len(a) > len(b) else "chosen_shorter" if len(a) < len(b) else "equal_length"
        groups[label].append(win)
    return {"accuracy": sum(wins) / 64, "per_label": {label: sum(v) / len(v) for label, v in groups.items()},
            "per_label_counts": {k: len(v) for k, v in groups.items()}, "pairs": 64,
            "per_label_scope": "source relevant-versus-irrelevant pairs grouped by answer length direction"}


def quality(formats):
    if set(formats) != {"title", "json", "prose"}:
        return {"passed": False, "failures": ["missing claimed input format"]}
    failures = []
    for name, value in formats.items():
        scalars = [value.get("accuracy"), value.get("untrained_accuracy"), value.get("shortcut_accuracy"), *value.get("per_label", {}).values()]
        if not value.get("per_label") or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in scalars):
            failures.append(name + " invalid measurement")
            continue
        if value["accuracy"] < .80 or min(value["per_label"].values()) < .70:
            failures.append(name + " accuracy/label floor")
        if value["accuracy"] + 1e-9 < value["untrained_accuracy"] + .05:
            failures.append(name + " untrained gain below five points")
        if value["shortcut_accuracy"] > .65 or value["accuracy"] + 1e-9 < value["shortcut_accuracy"] + .10:
            failures.append(name + " shortcut ceiling/gain failed")
    return {"passed": not failures, "failures": failures, "audit_pairs": 64,
            "all_supported_formats_tested": True, "formats": formats}


def ready(report):
    if report.get("family") != "ACADEMIC_SEARCH" or type(report.get("run_id")) is not int or report["run_id"] < 1 or not isinstance(report.get("commit"), str) or not re.fullmatch(r"[0-9a-f]{40}", report["commit"]):
        raise ValueError("real run/commit identity required")
    for name in ("package_sha256", "package_keccak"):
        if not isinstance(report.get(name), str) or not re.fullmatch(r"[0-9a-f]{64}", report[name]):
            raise ValueError("actual bare package hashes required")
    if report.get("serving_commit") != report["commit"]:
        raise ValueError("identical source serving commit required")
    source, runtime = report.get("source", {}), report.get("runtime", {})
    if (source.get("normalized_overlap") != 0 or source.get("audit_used_for_selection") is not False or source.get("grants_verified") is not True
            or source.get("independent_groups", 0) < 32 or not source.get("source_manifest_sha256")):
        raise ValueError("source gates not complete")
    if report.get("quality", {}).get("passed") is not True or not quality(report.get("quality", {}).get("formats", {}))["passed"]:
        raise ValueError("quality gates not complete")
    if (runtime.get("full_parser_passed") is not True or runtime.get("reference_engine_passed") is not True
            or runtime.get("repeat_bits_identical") is not True or runtime.get("negative_identity_controls", 0) < 3
            or not 8 <= len(runtime.get("vectors", [])) <= 100):
        raise ValueError("runtime gates not complete")
    for vector in runtime["vectors"]:
        score_value(vector["score_bits"])
        if not isinstance(vector.get("tokens"), list) or not 1 <= len(vector["tokens"]) <= 256 or any(type(t) is not int or not 0 <= t < 50368 for t in vector["tokens"]):
            raise ValueError("bounded actual token vector required")
    if (report.get("foundation_sha256"), report.get("engine_sha256"), report.get("tokenizer_sha256")) != (FOUNDATION, ENGINE, TOKENIZER):
        raise ValueError("pinned runtime identities required")
    return {**report, "schema_version": 1, "status": "REGISTRATION_READY"}
