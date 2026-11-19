# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""One-shot immutable package audit using actual reference-engine score bits."""
import argparse
from decimal import Decimal
import json
import math
import os
from pathlib import Path

from build import candidate, control_accuracy, sha, surface_accuracy, write_json
from core import render
from engine import executor
from protocol import bare_hash, decode_bits, negative_identities, request_value, score
from remote import CONTRACT, FAMILY, HERE, model_metadata, read_bound_splits, remote_only
import reference


def quality(formats, untrained, controls):
    failures = []
    if set(formats) != set(CONTRACT["views"]) or set(untrained) != set(CONTRACT["views"]) or not controls:
        return {"passed": False, "failures": ["missing view or shortcut"]}
    by_view = controls if set(controls) == set(CONTRACT["views"]) and all(isinstance(v, dict) for v in controls.values()) else {view: controls for view in CONTRACT["views"]}
    shortcut_numbers = [value for values in by_view.values() for value in values.values()]
    numbers = list(shortcut_numbers)
    for view in formats:
        numbers += [formats[view]["accuracy"], untrained[view]["accuracy"], *formats[view]["subsets"].values()]
    if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in numbers):
        return {"passed": False, "failures": ["invalid metric"]}
    if max(shortcut_numbers) > 0.65:
        failures.append("frozen shortcut exceeds 65 percent")
    for view, metrics in formats.items():
        if metrics["accuracy"] < 0.80:
            failures.append(view + " ordering below 80 percent")
        if metrics["accuracy"] + 1e-9 < untrained[view]["accuracy"] + 0.05:
            failures.append(view + " below five-point reference-untrained improvement")
        if metrics["accuracy"] + 1e-9 < max(by_view[view].values()) + 0.10:
            failures.append(view + " below ten-point frozen shortcut improvement")
        if set(metrics["subsets"]) != {"positive_error", "negative_error", "negative_c"} or any(x < 0.70 for x in metrics["subsets"].values()):
            failures.append(view + " incomplete or below 70 percent subset")
    return {"passed": not failures, "failures": failures}


def metrics(rows, scores):
    if len(scores) != len(rows) * 2:
        raise ValueError("score count mismatch")
    wins = [scores[2 * index] > scores[2 * index + 1] for index in range(len(rows))]
    def subset(predicate):
        selected = [win for row, win in zip(rows, wins) if predicate(row)]
        if not selected:
            raise ValueError("required subset absent")
        return sum(selected) / len(selected)
    return {"accuracy": sum(wins) / len(wins), "originals": len(rows),
            "subsets": {"positive_error": subset(lambda r: r["offset_tenths"] > 0),
                        "negative_error": subset(lambda r: r["offset_tenths"] < 0),
                        "negative_c": subset(lambda r: json.loads(r["chosen"])["temp_max_c"] < 0)}}


def identities(manifest):
    parser_hash = manifest["keccak256"]
    if not isinstance(parser_hash, str) or not parser_hash.startswith("0x"):
        raise ValueError("full parser did not emit its literal prefixed hash")
    return {"foundation_hash": CONTRACT["foundation"]["foundation_lora_fp32.onnx"],
            "tokenizer_hash": CONTRACT["foundation"]["tokenizer.json"], "adapter_hash": bare_hash(parser_hash[2:])}


def emit_ready(result, train_report, vectors, controls, source_report):
    if not result["passed"] or result["repeat_vectors"] < 8 or len(result["negative_identity_controls"]) < 3:
        raise ValueError("cannot emit readiness without actual passed audit/runtime controls")
    formats = {view: {"accuracy": value["accuracy"], "per_label": value["subsets"],
                      "untrained_accuracy": result["seed_matched_untrained"][view]["accuracy"],
                      "shortcut_accuracy": max(controls[view].values())} for view, value in result["formats"].items()}
    return {"schema_version": 1, "status": "REGISTRATION_READY", "family": FAMILY,
            "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
            "owner_attempt": CONTRACT["run_id"], "training_run_id": train_report["run_id"],
            "package_sha256": result["package_sha256"], "package_keccak": result["adapter_keccak"],
            "foundation_sha256": CONTRACT["foundation"]["foundation_lora_fp32.onnx"],
            "engine_sha256": CONTRACT["foundation"]["foundation_engine.wasm"],
            "tokenizer_sha256": CONTRACT["foundation"]["tokenizer.json"],
            "source": {"normalized_overlap": source_report["normalized_cross_split_overlap"], "audit_used_for_selection": False,
                       "grants_verified": True, "source_manifest_sha256": sha((HERE / "sources/capture-manifest.json").read_bytes()),
                       "independent_groups": source_report["splits"]["audit"]["originals"],
                       "data_sha256": sha((HERE / "data/data-report.json").read_bytes())},
            "quality": {"passed": result["passed"], "audit_pairs": source_report["splits"]["audit"]["originals"],
                        "all_supported_formats_tested": set(formats) == set(CONTRACT["views"]), "formats": formats},
            "runtime": {"full_parser_passed": result["full_parser_passed"], "reference_engine_passed": True,
                        "repeat_bits_identical": result["repeat_bits_identical"],
                        "negative_identity_controls": len(result["negative_identity_controls"]), "vectors": vectors},
            "serving_commit": os.environ["GITHUB_SHA"], "node_qualification": "not attempted"}


def audit(root):
    remote_only()
    if (root / "audit-start.json").exists() or (root / "audit-result.json").exists():
        raise ValueError("one-shot audit already started; preserve the existing evidence")
    train_report = json.loads((root / "result.json").read_text())
    package = root / (FAMILY + ".safetensors")
    untrained_package = root / "untrained.safetensors"
    if (train_report["commit"] != os.environ["GITHUB_SHA"] or train_report["package_sha256"] != sha(package.read_bytes())
            or train_report["untrained_package_sha256"] != sha(untrained_package.read_bytes())):
        raise ValueError("immutable candidate/untrained package or commit binding mismatch")
    data_report, splits, controls = read_bound_splits()
    if train_report["data"] != data_report:
        raise ValueError("train/audit source manifest mismatch")
    write_json(root / "audit-start.json", {**model_metadata(), "package_sha256": sha(package.read_bytes()), "policy": "one shot, no audit selection"})
    foundation = reference.foundation(root)
    kit = reference.source_kit(root)
    reference.compile_tools(root, kit)
    checked = reference.validate_package(root, package, "audit-parser.json")
    initial_checked = reference.validate_package(root, untrained_package, "audit-untrained-parser.json")
    from tokenizers import Tokenizer
    node = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    node.no_padding()
    node.no_truncation()
    names = json.loads((HERE / "data/station-names.json").read_text())
    vectors, view_rows = {}, {}
    for view in CONTRACT["views"]:
        vectors[view] = []
        view_rows[view] = []
        for row in splits["audit"]:
            responses = {}
            for side in ("chosen", "rejected"):
                obj = json.loads(row[side], parse_float=Decimal, parse_int=Decimal)
                answer = candidate(obj["temp_max_c"], names[row["station"]], row["day"], view)
                responses[side] = answer
                ids = node.encode(render(row["prompt"], answer)).ids
                if not ids or len(ids) > CONTRACT["hyperparameters"]["max_length"]:
                    raise ValueError("decisive audit claim truncated or empty")
                vectors[view].append(ids)
            view_rows[view].append({**row, **responses})
    formats, initial, fixed, negatives = {}, {}, [], []
    repeats = 0
    for label, candidate_package, manifest, output in (("untrained", untrained_package, initial_checked, initial),
                                                      ("trained", package, checked, formats)):
        identity = identities(manifest)
        with executor(root, candidate_package, label) as url:
            if label == "trained":
                negatives = negative_identities(url, vectors["json"][:2], identity)
                for tokens in ([], [-1]):
                    refusal = request_value(url, [tokens], identity)
                    if not refusal.get("results") or "error" not in refusal["results"][0]:
                        raise ValueError("actual executor failed token refusal control")
            for view, sequences in vectors.items():
                all_bits = []
                for offset in range(0, len(sequences), 4):
                    batch = sequences[offset:offset + 4]
                    first, second = score(url, batch, identity), score(url, batch, identity)
                    if first != second:
                        raise ValueError("actual reference score bits changed on identical inputs")
                    if label == "trained":
                        repeats += len(batch)
                        if view == "json" and len(fixed) < 16:
                            fixed.extend({"tokens": tokens, "score_bits": bit} for tokens, bit in zip(batch, first))
                    all_bits.extend(first)
                output[view] = metrics(splits["audit"], [decode_bits(bit) for bit in all_bits])
    frozen = {}
    for view, rows in view_rows.items():
        frozen[view] = {k: v["accuracy"] for k, v in control_accuracy(rows, controls["integer_coefficients"]).items()}
        frozen[view]["answer_only_bag_of_words"] = surface_accuracy(rows, controls["answer_only_token_weights"])["accuracy"]
    result = {**model_metadata(), "package_sha256": sha(package.read_bytes()), "adapter_keccak": identities(checked)["adapter_hash"],
              "formats": formats, "seed_matched_untrained": initial, "frozen_shortcuts": frozen,
              "full_parser_passed": checked["manifest"]["verification"]["stage1"], "repeat_bits_identical": True,
              "repeat_vectors": repeats, "negative_identity_controls": negatives,
              "reference_executable_sha256": sha((root / "adapterexec").read_bytes()), **quality(formats, initial, frozen)}
    write_json(root / "audit-result.json", result)
    if not result["passed"]:
        raise ValueError("independent actual reference-runtime quality gate failed")
    ready = emit_ready(result, train_report, fixed, frozen, data_report)
    ready["runtime"]["foundation_spec"] = json.loads((root / "foundation.json").read_text())
    from ready_contract import validate_ready
    validate_ready(ready)
    write_json(root / "registration-ready.json", ready)


if __name__ == "__main__":
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    audit(parser.parse_args().root)
