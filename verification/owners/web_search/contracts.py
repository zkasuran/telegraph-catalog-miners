# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Frozen source, literal wire and quality gates. No neural imports."""
import hashlib
import json
import math
import re
import struct
from collections import Counter
from fractions import Fraction
from pathlib import Path

from build import validate
from common import FOUNDATION, strict_json

HERE = Path(__file__).resolve().parent
FAMILY = "WEB_SEARCH"
REPO = "zkasuran/telegraph-catalog-miners"
HASH = re.compile(r"[a-f0-9]{64}")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def bounded(path, maximum=4 << 20):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("bounded regular artifact required")
    return path.read_bytes()


def read_splits():
    recipe = strict_json(bounded(HERE / "recipe.json"))
    raw_report = bounded(HERE / "source-data.json")
    if sha(raw_report) != recipe["source_manifest_sha256"]:
        raise ValueError("original complete source inventory changed")
    report = strict_json(raw_report)
    splits = {}
    for split in ("train", "validation", "audit"):
        raw = bounded(HERE / (split + ".jsonl"))
        if sha(raw) != recipe["files"][split]["sha256"] or recipe["files"][split] != report["files"][split]:
            raise ValueError("frozen split identity mismatch")
        splits[split] = [strict_json(line) for line in raw.split(b"\n") if line]
        if len(splits[split]) != recipe["files"][split]["rows"]:
            raise ValueError("frozen split count mismatch")
    validate(splits)
    if len({row["group_id"] for row in splits["audit"]}) < 32:
        raise ValueError("audit independent page gate")
    if sha(bounded(HERE / "baselines.json")) != recipe["baseline_sha256"]:
        raise ValueError("immutable train-fitted baseline changed")
    return recipe, report, splits


def orientation(row):
    query = strict_json(row["prompt"])["query"]
    match = re.fullmatch(r"Which (?:Python|Node\.js) version added (.+)\?", query)
    if match is None:
        raise ValueError("source query shape changed")
    if match[1] == row["facts"][0]["entity"]:
        return "target_first"
    if match[1] == row["facts"][1]["entity"]:
        return "target_second"
    raise ValueError("question is not bound to a source fact")


def shortcuts(rows):
    frozen = strict_json(bounded(HERE / "baselines.json"))
    scores = {"length": [], "answer_only": [], "overlap": []}
    for row in rows:
        length = len(row["chosen"]) - len(row["rejected"])
        words = set(re.findall(r"[a-z0-9']+", row["prompt"].casefold()))
        def overlap(side):
            return sum((Fraction(1, frozen["train_document_frequency"].get(word, 0) + 1)
                        for word in re.findall(r"[a-z0-9']+", row[side].casefold()) if word in words), Fraction(0))
        difference = overlap("chosen") - overlap("rejected")
        scores["length"].append(1 if length > 0 else .5 if length == 0 else 0)
        scores["answer_only"].append(.5)  # exact immutable zero coefficients
        scores["overlap"].append(1 if difference > 0 else .5 if difference == 0 else 0)
    return {key: sum(values) / len(values) for key, values in scores.items()}


def source_gate():
    recipe, report, splits = read_splits()
    expected = {"max_length": 256, "batch_size": 1, "grad_accum": 16, "epochs": 3,
                "eval_batch": 1, "lora_rank": 8, "lora_alpha": 32, "seed": 42}
    if recipe["family"] != FAMILY or recipe["formats"] != ["raw_json"] or recipe["hyperparameters"] != expected:
        raise ValueError("frozen family, format or CPU recipe mismatch")
    gradient = Counter()
    for row in splits["train"]:
        for side, sign in (("chosen", 1), ("rejected", -1)):
            for word in re.findall(r"[a-z0-9']+", row[side].casefold()):
                gradient[word] += sign
    if any(gradient.values()):
        raise ValueError("train-only exact zero baseline gradient failed")
    control = shortcuts(splits["validation"])
    if any(value > .65 for value in control.values()):
        raise ValueError("unchanged validation shortcut gate")
    for name, licence, expected_hash in (("python", "UPSTREAM-PSF.txt", report["sources"]["python"]["licence"]),
                                         ("node", "UPSTREAM-NODE.txt", report["sources"]["node"]["licence"])):
        if expected_hash not in ("PSF-2.0", "MIT") or not bounded(HERE / licence):
            raise ValueError("source grant absent: " + name)
    for name, expected in {"UPSTREAM-PSF.txt": "b0e25a78cffb43f4d92de8b61ccfa1f1f98ecbc22330b54b5251e7b6ba010231",
                           "UPSTREAM-NODE.txt": "b58bb782ec115892f61475a6199db165545926d93624ae172f25b7a6e1dafffe"}.items():
        if sha(bounded(HERE / name)) != expected:
            raise ValueError("reviewed original source grant changed")
    return {"pairs": sum(map(len, splits.values())), "independent_audit_pages": recipe["files"]["audit"]["pages"],
            "validation_shortcuts": control, "normalized_overlap": 0, "audit_metrics": "sealed"}


def finite_bits(value):
    if type(value) is not int or not 0 <= value < 1 << 32:
        raise ValueError("unsigned score bits required")
    number = struct.unpack("<f", struct.pack("<I", value))[0]
    if not math.isfinite(number):
        raise ValueError("nonfinite score")
    return number


def literal_reply(value, count, keccak, root=None):
    expected = {"adapter_hash": keccak, "foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
                "tokenizer_hash": FOUNDATION["tokenizer.json"]}
    if not isinstance(value, dict) or any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("literal bare-hash reply identity mismatch")
    if root and value.get("input_root") != root:
        raise ValueError("literal input root mismatch")
    rows = value.get("results")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("literal result count mismatch")
    scores = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"score"}:
            raise ValueError("literal result shape mismatch")
        finite_bits(row["score"])
        scores.append(row["score"])
    return scores


def metrics(rows, scores):
    if len(scores) != 2 * len(rows) or not rows:
        raise ValueError("actual pair score count mismatch")
    wins = [finite_bits(scores[2 * i]) > finite_bits(scores[2 * i + 1]) for i in range(len(rows))]
    labels = {orientation(row) for row in rows}
    per_label = {label: sum(win for row, win in zip(rows, wins) if orientation(row) == label) /
                 sum(orientation(row) == label for row in rows) for label in labels}
    pages = {row["group_id"] for row in rows}
    per_page = {page: sum(win for row, win in zip(rows, wins) if row["group_id"] == page) /
                sum(row["group_id"] == page for row in rows) for page in pages}
    return {"accuracy": sum(wins) / len(rows), "per_label": per_label, "pairs": len(rows),
            "independent_pages": len(pages), "page_macro_accuracy": sum(per_page.values()) / len(per_page)}


def quality_gate(trained, untrained_accuracy, shortcut_accuracy):
    values = [trained["accuracy"], untrained_accuracy, shortcut_accuracy, *trained["per_label"].values()]
    if not trained["per_label"] or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("finite actual quality metrics required")
    return (trained["accuracy"] >= .8 and min(trained["per_label"].values()) >= .7 and
            trained["accuracy"] + 1e-9 >= untrained_accuracy + .05 and
            trained["accuracy"] + 1e-9 >= shortcut_accuracy + .10)


def ready_report(report):
    if (not isinstance(report, dict) or type(report.get("schema_version")) is not int or report.get("schema_version") != 1 or
            report.get("status") != "REGISTRATION_READY" or report.get("family") != FAMILY):
        raise ValueError("registration-ready envelope mismatch")
    if type(report.get("run_id")) is not int or report["run_id"] < 1:
        raise ValueError("actual positive GitHub run identity required")
    for name in ("commit", "serving_commit"):
        if not isinstance(report.get(name), str) or not re.fullmatch(r"[a-f0-9]{40}", report[name]):
            raise ValueError("literal source commit identity required")
    for name in ("package_sha256", "package_keccak"):
        if not isinstance(report.get(name), str) or not HASH.fullmatch(report[name]):
            raise ValueError("literal full lowercase package identity required")
    if type(report.get("package_bytes")) is not int or not 8 < report["package_bytes"] <= 64 << 20:
        raise ValueError("exact bounded package byte identity required")
    if "package_manifest" in report:
        manifest = report["package_manifest"]
        if (not isinstance(manifest, dict) or manifest.get("sha256") != report["package_sha256"] or
                manifest.get("keccak256") != "0x" + report["package_keccak"] or
                type(manifest.get("bytes")) is not int or manifest["bytes"] != report["package_bytes"]):
            raise ValueError("primary parser manifest and report identities differ")
    for name, expected in (("foundation_sha256", FOUNDATION["foundation_lora_fp32.onnx"]),
                           ("engine_sha256", FOUNDATION["foundation_engine.wasm"]),
                           ("tokenizer_sha256", FOUNDATION["tokenizer.json"])):
        if report.get(name) != expected:
            raise ValueError("pinned runtime identity mismatch")
    source, quality, runtime = (report.get(name) for name in ("source", "quality", "runtime"))
    if not all(isinstance(section, dict) for section in (source, quality, runtime)):
        raise ValueError("source, quality and runtime report objects required")
    if (type(source.get("normalized_overlap")) is not int or source.get("normalized_overlap") != 0 or
            source.get("audit_used_for_selection") is not False or source.get("grants_verified") is not True or
            type(source.get("independent_groups")) is not int or source["independent_groups"] < 32 or
            not isinstance(source.get("source_manifest_sha256"), str) or not HASH.fullmatch(source["source_manifest_sha256"])):
        raise ValueError("source and independent audit gate")
    recipe_raw = bounded(HERE / "recipe.json")
    recipe = strict_json(recipe_raw)
    if (source["source_manifest_sha256"] != recipe["source_manifest_sha256"] or
            ("recipe_sha256" in report and report["recipe_sha256"] != sha(recipe_raw))):
        raise ValueError("report differs from frozen training source or recipe")
    if (quality.get("passed") is not True or type(quality.get("audit_pairs")) is not int or quality["audit_pairs"] < 64 or
            quality.get("all_supported_formats_tested") is not True or not isinstance(quality.get("formats"), dict) or
            set(quality["formats"]) != {"raw_json"}):
        raise ValueError("complete supported format audit required")
    for measured in quality["formats"].values():
        if (not isinstance(measured, dict) or not {"accuracy", "untrained_accuracy", "shortcut_accuracy", "per_label"} <= set(measured) or
                not isinstance(measured["per_label"], dict) or set(measured["per_label"]) != {"target_first", "target_second"}):
            raise ValueError("both actual target-position labels and baselines required")
        if not quality_gate(measured, measured["untrained_accuracy"], measured["shortcut_accuracy"]):
            raise ValueError("unchanged quality floor failed")
    if (runtime.get("full_parser_passed") is not True or runtime.get("reference_engine_passed") is not True or
            runtime.get("repeat_bits_identical") is not True or type(runtime.get("negative_identity_controls")) is not int or
            runtime["negative_identity_controls"] < 3 or not isinstance(runtime.get("vectors"), list) or
            not 8 <= len(runtime["vectors"]) <= 100):
        raise ValueError("actual parser, engine, repeat and identity proof required")
    for vector in runtime["vectors"]:
        if not isinstance(vector, dict) or set(vector) != {"tokens", "score_bits"}:
            raise ValueError("literal fixed reference vector shape required")
        finite_bits(vector["score_bits"])
        tokens = vector["tokens"]
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= 256 or not all(type(token) is int and 0 <= token < 50368 for token in tokens):
            raise ValueError("fixed reference token identity or cap")
    return report
