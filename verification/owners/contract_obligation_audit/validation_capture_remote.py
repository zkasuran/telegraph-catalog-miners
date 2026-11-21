# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Frozen validation-only reference-engine capture. Never an independent audit."""
import argparse
from collections import Counter
import os
from pathlib import Path
import re
import time

from audit_remote import request
from contracts import FAMILY, FORMATS, FOUNDATION, HERE, check_config, check_wire, decode_bits, file_sha, remote_only, render, strict_json, write_json
from development_diagnostics import checked_pair, load_rows, wire_diagnostics
from executor import ReferenceExecutor, validate_request
from prepare_remote import compile_runtime, full_parser, prepare

MAX_METADATA_BYTES = 2 << 20
MAX_VALIDATION_PAIRS = 30
CAPTURE_SECONDS = 1200


def metadata(path):
    if path.is_symlink() or path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("validation metadata byte ceiling")
    value = strict_json(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("validation metadata object required")
    return value


def verify_freeze(freeze, source, source_hash, validation_hash, run_id, commit):
    if (freeze.get("schema_version") != 1 or freeze.get("family") != FAMILY
            or type(run_id) is not int or run_id < 1 or type(freeze.get("run_id")) is not int or freeze.get("run_id") != run_id
            or not isinstance(commit, str) or not re.fullmatch(r"[a-f0-9]{40}", commit) or freeze.get("commit") != commit
            or freeze.get("audit_used_for_selection") is not False
            or freeze.get("source_manifest_sha256") != source_hash):
        raise ValueError("actual immutable training freeze identity mismatch")
    files = source.get("files")
    if (not isinstance(files, dict) or not isinstance(files.get("validation"), dict)
            or files["validation"].get("sha256") != validation_hash):
        raise ValueError("frozen validation source identity mismatch")
    for key in ("candidate_sha256", "untrained_sha256"):
        if not isinstance(freeze.get(key), str) or not re.fullmatch(r"[a-f0-9]{64}", freeze[key]):
            raise ValueError("actual frozen package identity required")


def validation_vectors(rows, encoder):
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_VALIDATION_PAIRS:
        raise ValueError("bounded validation originals required")
    vectors = {}
    for style in FORMATS:
        vectors[style] = []
        for row in rows:
            view = checked_pair(row, style, "validation")
            for side in ("chosen", "rejected"):
                ids = encoder(render(view["prompt"], view[side]))
                if (not isinstance(ids, list) or not 1 <= len(ids) <= 512
                        or any(type(token) is not int or not 0 <= token < 50368 for token in ids)):
                    raise ValueError("validation claim would truncate or token vector is invalid")
                vectors[style].append(ids)
    return vectors


def capture_batches(score, vectors, package_keccak, deadline, clock=time.monotonic):
    """Pure orchestration boundary; local tests supply authored wire fixtures."""
    if not isinstance(vectors, dict) or set(vectors) != set(FORMATS):
        raise ValueError("all validation wire formats required")
    captured = {}
    for style in FORMATS:
        tokens = vectors[style]
        if not isinstance(tokens, list) or not 2 <= len(tokens) <= MAX_VALIDATION_PAIRS * 2 or len(tokens) % 2:
            raise ValueError("paired validation vector ceiling")
        combined = None
        for start in range(0, len(tokens), 8):
            remaining = deadline - clock()
            if remaining <= 1:
                raise TimeoutError("frozen validation capture deadline")
            batch = tokens[start:start + 8]
            payload = request(batch, package_keccak)
            validate_request(payload, package_keccak)
            response = score(payload, timeout=min(600, remaining))
            if (not isinstance(response, dict) or not isinstance(response.get("results"), list)
                    or any(not isinstance(item, dict) or "score" not in item for item in response["results"])):
                raise ValueError("malformed reference validation response")
            check_wire(response, len(batch), package_keccak)
            if combined is None:
                # Copy actual returned echoes, never fill results from constants.
                combined = {key: response[key] for key in ("foundation_hash", "adapter_hash", "tokenizer_hash")}
                combined["results"] = []
            combined["results"].extend({"score": item["score"]} for item in response["results"])
        check_wire(combined, len(tokens), package_keccak)
        captured[style] = combined
    return captured


def summarize(rows, responses, package_keccak):
    observed = wire_diagnostics(rows, responses, package_keccak)
    for style in FORMATS:
        bits = check_wire(responses[style], len(rows) * 2, package_keccak)
        margins = [decode_bits(bits[2 * index]) - decode_bits(bits[2 * index + 1]) for index in range(len(rows))]
        by_hypothesis = {}
        for key in sorted({row["hypothesis_id"] for row in rows}):
            selected = [margins[index] for index, row in enumerate(rows) if row["hypothesis_id"] == key]
            by_hypothesis[key] = {"pairs": len(selected), "strict_correct": sum(value > 0 for value in selected),
                                  "ties": sum(value == 0 for value in selected)}
        observed["formats"][style]["per_hypothesis"] = by_hypothesis
    return observed


def capture(root):
    remote_only()
    started = time.monotonic()
    config = check_config(metadata(HERE / "config.json"))
    freeze_path, source_path = root / "training-freeze.json", root / "data/source-manifest.json"
    validation_path = root / "data/validation.jsonl"
    freeze, source = metadata(freeze_path), metadata(source_path)
    run_id, commit = int(os.environ["GITHUB_RUN_ID"]), os.environ["GITHUB_SHA"]
    rows = load_rows(validation_path, "validation")
    freeze_hash, source_hash, validation_hash = file_sha(freeze_path), file_sha(source_path), file_sha(validation_path)
    verify_freeze(freeze, source, source_hash, validation_hash, run_id, commit)
    if len(rows) != config["limits"]["validation"] or set(Counter(row["label"] for row in rows)) != {"Entailment", "Contradiction"}:
        raise ValueError("actual frozen validation population differs")
    foundation, _, kit = prepare(root)
    compile_runtime(root, kit)
    actual_foundation = {name: file_sha(foundation / name) for name in FOUNDATION}
    if actual_foundation != FOUNDATION:
        raise ValueError("actual reference foundation/engine/tokenizer identity mismatch")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    vectors = validation_vectors(rows, lambda text: tokenizer.encode(text).ids)
    deadline = started + CAPTURE_SECONDS
    models, raw_wires = {}, {}
    for name, expected in (("untrained", freeze["untrained_sha256"]), (FAMILY, freeze["candidate_sha256"])):
        if deadline - time.monotonic() <= 1:
            raise TimeoutError("frozen validation capture deadline before model startup")
        package = root / (name + ".safetensors")
        if package.is_symlink() or not 8 < package.stat().st_size <= 64 << 20:
            raise ValueError("bounded frozen package file required")
        if file_sha(package) != expected:
            raise ValueError("frozen validation package bytes changed")
        parsed = full_parser(root, package, root / "foundation.json")
        package_keccak = parsed["keccak256"].removeprefix("0x")
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            raise TimeoutError("frozen validation capture deadline before model startup")
        backend = ReferenceExecutor(root, package, package_keccak, startup_timeout=min(180, remaining))
        try:
            responses = capture_batches(backend.score, vectors, package_keccak, deadline)
        finally:
            backend.close()
        if file_sha(package) != expected:
            raise ValueError("frozen package changed during validation capture")
        raw_wires[name] = responses
        observed = summarize(rows, responses, package_keccak)
        observed["source_of_scores"] = "actual pinned ReferenceExecutor in this public workflow run"
        observed["new_inference_performed"] = True
        observed["limits"] = ["Actual reference-engine validation responses, not an independent audit.",
                              "This capture does not select a model or establish registration-ready status."]
        models[name] = {"package_sha256": file_sha(package), "package_keccak": package_keccak,
                        "full_parser_report_sha256": file_sha(root / (package.stem + "-parser.json")),
                        "observed_validation": observed}
        write_json(root / "validation-runtime-wire.json", raw_wires)
    if (file_sha(freeze_path) != freeze_hash or file_sha(source_path) != source_hash
            or file_sha(validation_path) != validation_hash):
        raise ValueError("frozen source or candidate selection changed during capture")
    report = {"schema_version": 1, "status": "VALIDATION_RUNTIME_OBSERVED", "family": FAMILY,
              "run_id": run_id, "commit": commit, "source_manifest_sha256": file_sha(source_path),
              "validation_data_sha256": file_sha(validation_path), "training_freeze_sha256": file_sha(freeze_path),
              "foundation_sha256": actual_foundation["foundation_lora_fp32.onnx"],
              "engine_sha256": actual_foundation["foundation_engine.wasm"], "tokenizer_sha256": actual_foundation["tokenizer.json"],
              "executor_sha256": file_sha(root / "adapterexec"), "wire_capture_sha256": file_sha(root / "validation-runtime-wire.json"),
              "audit_rows_read": False, "independent_audit": False, "candidate_selection_performed": False,
              "new_reference_inference_performed": True,
              "model_or_data_changed": False, "fit_role": "frozen_validation_only", "models": models,
              "limits": ["Validation source is development material, not an independent audit.",
                         "No registration-ready status, fixed-vector qualification or canonical win is established by this capture."]}
    write_json(root / "validation-runtime-report.json", report)
    print("Actual frozen validation reference-wire responses captured. Independent audit was not read.")
    return report


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return capture(parser.parse_args().root.resolve())


if __name__ == "__main__":
    main()
