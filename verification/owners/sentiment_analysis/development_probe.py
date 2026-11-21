# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Validation-only task alignment controls on the frozen reference scorer."""
import argparse
import copy
import http.client
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

from common import json_bytes
from contracts import FAMILY, FORMATS, FOUNDATION, FROZEN, HERE, OWNER, fetch, file_sha, finite_bits, remote_only, sha, strict_json, validate_ready
from remote import assets, executor, identity_controls, score, stage_check
from retrain_data import SOURCE_FILE, download
from source_data import group_id, inventory
from v3_data import BASELINE, LABELS, QUARANTINE, baselines, dump, pairs, select_groups, unanimous_inventory

CONFIG = HERE / "development-config.json"
PACKAGE = HERE / "model/SENTIMENT_ANALYSIS-endpoint-recovery.safetensors"
CONDITIONS = ("aligned", "answer_only", "cross_label_prompt")


def check_config(value):
    if (not isinstance(value, dict) or value.get("schema_version") != 1 or value.get("family") != FAMILY or value.get("split") != "validation"
            or value.get("conditions") != list(CONDITIONS) or type(value.get("per_label_groups")) is not int
            or not 1 <= value["per_label_groups"] <= 16 or value.get("max_tokens") != 256):
        raise ValueError("bounded validation-only configuration required")
    for key in ("validation_groups_sha256", "sources_sha256", "baseline_sha256", "quarantine_sha256", "package_sha256", "package_keccak", "cases_sha256"):
        if not isinstance(value.get(key), str) or not re.fullmatch("[a-f0-9]{64}", value[key]):
            raise ValueError("actual frozen source/model hash required")
    return value


def config():
    return check_config(strict_json(CONFIG.read_bytes()))


def source(root, raw_root=None):
    """Inspect the full legitimate annotation release, then export validation only."""
    cfg = config()
    root.mkdir(parents=True, exist_ok=True)
    raw_root = raw_root or root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    sources = strict_json(SOURCE_FILE.read_bytes())
    if sources.get("stage") != "verified" or file_sha(SOURCE_FILE) != cfg["sources_sha256"]:
        raise ValueError("original complete source grants/hash required")
    if file_sha(BASELINE) != cfg["baseline_sha256"] or file_sha(QUARANTINE) != cfg["quarantine_sha256"]:
        raise ValueError("original train-fitted baseline or quarantine changed")
    for name, spec in sources["files"].items():
        download(spec["url"], raw_root / name, spec["sha256"])
    originals, inventory_report = unanimous_inventory(raw_root, strict_json((raw_root / "sentiment_mapping.json").read_bytes()))
    excluded = root / "excluded-uci.zip"
    fetch(FROZEN["source_url"], excluded, FROZEN["source_sha256"], 1 << 20)
    uci, _ = inventory(excluded)
    grouped, selection = select_groups(originals, {group_id(row["text"]) for row in uci})
    validation = grouped["validation"]
    del grouped
    dump(root / "validation-groups.json", validation)
    if file_sha(root / "validation-groups.json") != cfg["validation_groups_sha256"]:
        raise ValueError("original legitimate validation groups changed")
    report = {"schema_version": 1, "family": FAMILY, "split": "validation", "inventory": inventory_report,
              "validation_groups_sha256": cfg["validation_groups_sha256"], "sources_sha256": cfg["sources_sha256"],
              "baseline_sha256": cfg["baseline_sha256"], "quarantine_sha256": cfg["quarantine_sha256"],
              "normalized_overlap": 0, "audit_rows_scored": 0, "fitting": False,
              "selection_exclusions": selection["excluded"], "grants_verified": True,
              "construction": "Upstream emotion annotations with programmatic valence and composition preferences, not human sentiment-preference review."}
    dump(root / "development-source.json", report)
    return validation, report


def cases(validation, per_label):
    if type(per_label) is not int or not 1 <= per_label <= 16:
        raise ValueError("bounded deterministic group selection required")
    selected = []
    for label in LABELS:
        candidates = sorted((row for row in validation if row["label"] == label), key=lambda row: row["group_id"])
        if len(candidates) < per_label:
            raise ValueError("missing development label coverage")
        selected += candidates[:per_label]
    if len({row["group_id"] for row in selected}) != len(selected):
        raise ValueError("duplicate validation source group")
    output = {condition: {} for condition in CONDITIONS}
    for style, name in enumerate(FORMATS):
        aligned = pairs(selected, style)
        representative = {(label, row["group_id"]): row["prompt"] for row in aligned for label in [row["label"]]}
        by_label = {label: sorted(((gid, prompt) for (kind, gid), prompt in representative.items() if kind == label)) for label in LABELS}
        shuffled = []
        for row in aligned:
            other_label = LABELS[(LABELS.index(row["label"]) + 1) % len(LABELS)]
            index = next(i for i, (gid, _) in enumerate(by_label[row["label"]]) if gid == row["group_id"])
            other_group, other_prompt = by_label[other_label][index]
            shuffled.append({**row, "prompt": other_prompt, "prompt_group_id": other_group, "prompt_label": other_label})
        output["aligned"][name] = aligned
        output["answer_only"][name] = [{**row, "prompt": ""} for row in aligned]
        output["cross_label_prompt"][name] = shuffled
    return output


def render(row, side):
    # Literal primary train_adapter.render contract. No tokenizer or neural import locally.
    return f"### Task:\n{row['prompt']}\n\n### Answer:\n{row[side]}"


def measure(rows, bits, coefficient):
    if len(bits) != 2 * len(rows) or not rows or Counter(row["label"] for row in rows).keys() != set(LABELS):
        raise ValueError("complete finite score pairs and every label required")
    values = [finite_bits(value) for value in bits]
    wins = [values[2*i] > values[2*i+1] for i in range(len(rows))]
    ties = sum(values[2*i] == values[2*i+1] for i in range(len(rows)))
    return {"pair_preference_rate": sum(wins)/len(rows), "pairs": len(rows), "ties": ties,
            "per_label": {label: sum(win for row, win in zip(rows, wins) if row["label"] == label) /
                                  sum(row["label"] == label for row in rows) for label in LABELS},
            "shortcuts": baselines(rows, coefficient),
            "utf8_length_range": [min(len(render(row, side).encode()) for row in rows for side in ("chosen", "rejected")),
                                  max(len(render(row, side).encode()) for row in rows for side in ("chosen", "rejected"))]}


def validate_diagnostic(report):
    cfg = config()
    if (not isinstance(report, dict) or report.get("schema_version") != 1 or report.get("status") != "DEVELOPMENT_DIAGNOSTIC"
            or report.get("family") != FAMILY or report.get("owner") != OWNER or type(report.get("run_id")) is not int
            or report["run_id"] < 1 or not re.fullmatch("[a-f0-9]{40}", report.get("commit", ""))
            or report.get("model_fitted") is not False or report.get("candidate_selected") is not False
            or type(report.get("audit_rows_scored")) is not int or report["audit_rows_scored"] != 0
            or report.get("independent_groups") != cfg["per_label_groups"] * 4 or report.get("cases_sha256") != cfg["cases_sha256"]):
        raise ValueError("actual validation-only diagnostic identity/boundary required")
    for key, expected in (("package_sha256", cfg["package_sha256"]), ("package_keccak", cfg["package_keccak"]),
                          ("config_sha256", file_sha(CONFIG)), ("foundation_sha256", FOUNDATION["foundation_lora_fp32.onnx"]),
                          ("engine_sha256", FOUNDATION["foundation_engine.wasm"]), ("tokenizer_sha256", FOUNDATION["tokenizer.json"])):
        if report.get(key) != expected:
            raise ValueError("frozen candidate/runtime/config identity changed")
    source_report = report.get("source", {})
    if (source_report.get("split") != "validation" or source_report.get("validation_groups_sha256") != cfg["validation_groups_sha256"]
            or source_report.get("baseline_sha256") != cfg["baseline_sha256"] or source_report.get("normalized_overlap") != 0
            or source_report.get("grants_verified") is not True or source_report.get("audit_rows_scored") != 0):
        raise ValueError("original source and immutable training baseline required")
    metrics = report.get("metrics", {})
    if set(metrics) != set(CONDITIONS) or any(set(formats) != set(FORMATS) for formats in metrics.values()):
        raise ValueError("every diagnostic condition and response format required")
    for formats in metrics.values():
        for values in formats.values():
            if set(values.get("per_label", {})) != set(LABELS) or values.get("pairs") != cfg["per_label_groups"] * 12:
                raise ValueError("every label and actual contrast count required")
            fractions = [values.get("pair_preference_rate"), *values["per_label"].values(), *values.get("shortcuts", {}).values()]
            if not values.get("shortcuts") or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in fractions):
                raise ValueError("finite actual diagnostic/shortcut measurements required")
            if type(values.get("ties")) is not int or not 0 <= values["ties"] <= values["pairs"]:
                raise ValueError("actual tie count required")
    runtime = report.get("runtime", {})
    if any(runtime.get(name) is not True for name in ("full_parser_passed", "reference_engine_passed", "repeat_bits_identical", "literal_wire_passed")) or runtime.get("negative_identity_controls", 0) < 3:
        raise ValueError("actual parser/reference/literal controls required")
    if not 8 <= len(runtime.get("vectors", [])) <= 100:
        raise ValueError("actual finite reference vectors required")
    for item in runtime["vectors"]:
        finite_bits(item["score_bits"])
    for key in ("cases_sha256", "score_bits_sha256"):
        if not re.fullmatch("[a-f0-9]{64}", report.get(key, "")):
            raise ValueError("actual data and bit report hashes required")
    return report


def literal_wire(server, identity_row, vectors, bits):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=180)
    body = {"batch": [{"tokens": ids} for ids in vectors], "foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
            "adapter_hash": identity_row["keccak"], "tokenizer_hash": FOUNDATION["tokenizer.json"]}
    try:
        connection.request("POST", "/runnable", json_bytes(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read(65537)
        result = strict_json(raw)
        if len(raw) > 65536 or response.status != 200 or any(result.get(k) != body[k] for k in ("foundation_hash", "adapter_hash", "tokenizer_hash")) or result.get("results") != [{"score": value} for value in bits]:
            raise ValueError("actual literal bare-hash wire or score-bit contract failed")
    finally:
        connection.close()


def remote(root):
    remote_only()
    cfg = config()
    if file_sha(PACKAGE) != cfg["package_sha256"]:
        raise ValueError("exact frozen development model changed")
    validation, source_report = source(root / "data")
    views = cases(validation, cfg["per_label_groups"])
    cases_sha = sha(json_bytes(views))
    if cases_sha != cfg["cases_sha256"]:
        raise ValueError("pre-dispatch frozen validation cases changed")
    dump(root / "development-input-identity.json", {"cases_sha256": cases_sha, "groups_sha256": cfg["validation_groups_sha256"], "split": "validation"})
    assets(root)
    parsed = stage_check(root, PACKAGE, cfg["package_sha256"], "development-parser")
    if parsed["keccak256"].removeprefix("0x") != cfg["package_keccak"]:
        raise ValueError("actual full-parser package Keccak changed")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(root / "foundation/tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    tokens = {condition: {name: [tokenizer.encode(render(row, side)).ids for row in rows for side in ("chosen", "rejected")]
                          for name, rows in formats.items()} for condition, formats in views.items()}
    if any(not 1 <= len(ids) <= cfg["max_tokens"] for formats in tokens.values() for batch in formats.values() for ids in batch):
        raise ValueError("development text exceeds frozen token limit; truncation not allowed")
    origin = validate_ready(strict_json((HERE / "model/endpoint-registration-ready.json").read_bytes()))
    vectors = [item["tokens"] for item in origin["runtime"]["vectors"]]
    expected = [item["score_bits"] for item in origin["runtime"]["vectors"]]
    coefficient = strict_json(BASELINE.read_bytes())
    metrics, actual_bits = {}, {}
    with executor(root, PACKAGE, cfg["package_keccak"]) as (backend, identity_row, server):
        negatives = identity_controls(server, identity_row)
        first, second = score(backend, identity_row, vectors), score(backend, identity_row, vectors)
        if first != expected or second != expected:
            raise ValueError("frozen candidate reference vectors changed")
        literal_wire(server, identity_row, vectors, first)
        for condition, formats in views.items():
            metrics[condition], actual_bits[condition] = {}, {}
            for name, rows in formats.items():
                bits = score(backend, identity_row, tokens[condition][name])
                actual_bits[condition][name] = bits
                metrics[condition][name] = measure(rows, bits, coefficient)
                metrics[condition][name]["token_length_range"] = [min(map(len, tokens[condition][name])), max(map(len, tokens[condition][name]))]
    dump(root / "development-score-bits.json", actual_bits)
    report = {"schema_version": 1, "status": "DEVELOPMENT_DIAGNOSTIC", "family": FAMILY, "owner": OWNER,
              "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
              "config_sha256": file_sha(CONFIG), "package_sha256": cfg["package_sha256"], "package_keccak": cfg["package_keccak"],
              "foundation_sha256": FOUNDATION["foundation_lora_fp32.onnx"], "engine_sha256": FOUNDATION["foundation_engine.wasm"],
              "tokenizer_sha256": FOUNDATION["tokenizer.json"], "source": source_report, "cases_sha256": cases_sha,
              "independent_groups": cfg["per_label_groups"]*len(LABELS), "audit_rows_scored": 0,
              "model_fitted": False, "candidate_selected": False, "metrics": metrics,
              "score_bits_sha256": file_sha(root / "development-score-bits.json"),
              "runtime": {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
                          "negative_identity_controls": negatives, "literal_wire_passed": True,
                          "vectors": copy.deepcopy(origin["runtime"]["vectors"])},
              "limits": ["Validation-only development measurements, not an independent audit or registration-ready result.",
                         "Prompt-permuted preference retention is not factual accuracy because the original answer label does not match the new task.",
                         "Answer-only JSON/prose retains source excerpts; this measures information leakage through answers, not a universal label-only shortcut.",
                         "Original model, audit and upstream grants remain unchanged. No new human sentiment-preference review is claimed."]}
    validate_diagnostic(report)
    dump(root / "development-report.json", report)
    print(json.dumps({"status": report["status"], "groups": report["independent_groups"], "metrics": metrics}, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("source", "remote"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    args = parser.parse_args()
    if args.phase == "source":
        validation, report = source(args.root, args.raw_root)
        built = cases(validation, config()["per_label_groups"])
        if sha(json_bytes(built)) != config()["cases_sha256"]:
            raise ValueError("pre-dispatch frozen validation cases changed")
        dump(args.root / "development-cases.json", built)
        print(json.dumps({"source": report, "cases_sha256": sha(json_bytes(built))}, sort_keys=True))
    else:
        remote_only()
        remote(args.root)
