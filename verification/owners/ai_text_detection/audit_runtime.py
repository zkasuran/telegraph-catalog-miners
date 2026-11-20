# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""One frozen audit in the actual foundation engine, including untrained control."""
import argparse
import contextlib
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path

from common import (COUNTS, FAMILY, FORMATS, HERE, MAX_TOKENS, SCOPE, baseline_report,
                    check_splits, dump, lines, pair, quality, remote_only, render, sha, identity)
from run import FOUNDATION
from wire import check_reply, request, score_value


@contextlib.contextmanager
def executor(root, package):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "FOUNDATION_MODEL_PATH": str(root / "foundation/foundation_lora_fp32.onnx"),
           "FOUNDATION_MODEL_HASH": FOUNDATION["foundation_lora_fp32.onnx"],
           "FOUNDATION_ENGINE_WASM_PATH": str(root / "foundation/foundation_engine.wasm"),
           "FOUNDATION_ENGINE_WASM_HASH": FOUNDATION["foundation_engine.wasm"]}
    with (root / (package.stem + "-runtime.log")).open("a") as log:
        process = subprocess.Popen([str(root / "adapterexec"), "--package", str(package),
                                    "--listen", f"127.0.0.1:{port}"], stdout=log, stderr=log, env=env)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("actual reference executor failed to load")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    time.sleep(1)
            else:
                raise RuntimeError("reference executor readiness timeout")
            yield f"http://127.0.0.1:{port}/runnable"
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)


def measures(rows, bits):
    wins = [score_value(bits[i*2]) > score_value(bits[i*2+1]) for i in range(len(rows))]
    per = {label: sum(w for r, w in zip(rows, wins) if r["label"] == label) /
           sum(r["label"] == label for r in rows) for label in ("Human", "AI-generated")}
    return {"n_eval": len(rows), "pairwise_accuracy": sum(wins) / len(wins), "accuracy_by_label": per}


def main():
    remote_only()
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    root = p.parse_args().root.resolve()
    if (root / "audit-runtime-result.json").exists():
        raise ValueError("audit already measured; preserve result and do not reroll")
    frozen = json.loads((root / "frozen-candidate.json").read_text())
    package = root / (FAMILY + ".safetensors")
    if sha(package.read_bytes()) != frozen["package_sha256"] or frozen["audit_used_for_selection"] is not False:
        raise ValueError("candidate changed after validation freeze")
    source_raw = (HERE / "sources.json").read_bytes()
    from supplement_contract import source_bundle, validate_generation
    from secondary_contract import expanded_source_bundle, validate_secondary
    bundle_sha, _, extra_sources, base_sources = expanded_source_bundle()
    if sha(source_raw) != frozen["base_source_manifest_sha256"] or bundle_sha != frozen["source_manifest_sha256"]:
        raise ValueError("source changed after candidate freeze")
    for path, key in (("frozen-baselines.json", "baseline_sha256"), ("development-generation.jsonl", "development_sha256")):
        if sha((root / path).read_bytes()) != frozen[key]:
            raise ValueError("development or baseline changed after freeze")
    records = lines((root / "development-generation.jsonl").read_bytes()) + lines((root / "audit-generation.jsonl").read_bytes())
    augmentation = (root / "augmentation-generation.jsonl").read_bytes()
    if sha(augmentation) != frozen["training_expansion"]["augmentation_sha256"]:
        raise ValueError("training-only augmentation changed after model freeze")
    additional = (root / "additional-augmentation-generation.jsonl").read_bytes()
    if sha(additional) != frozen["training_expansion"]["additional_augmentation_sha256"]:
        raise ValueError("additional train-only generation changed after model freeze")
    kept_views = set(frozen["training_expansion"]["kept_view_ids"])
    records += [r for r in lines(augmentation) + lines(additional) if r["view_id"] in kept_views]
    supplement_raw = (root / "supplement-generation.jsonl").read_bytes()
    if sha(supplement_raw) != frozen["supplement_generation_sha256"]:
        raise ValueError("source-only supplement changed after candidate freeze")
    supplement_report = json.loads((root / "supplement-generation-report.json").read_text())
    supplement_views = validate_generation(supplement_raw, supplement_report, extra_sources, base_sources, [])
    supplement_marks = {r["text_id"] for r in supplement_views} | {identity(r["text"]) for r in lines(supplement_raw)}
    if supplement_marks.intersection(identity(r["text"]) for r in records if r["split"] == "test"):
        raise ValueError("supplementary training overlaps independent audit")
    secondary_raw = (root / "secondary-generation.jsonl").read_bytes()
    if sha(secondary_raw) != frozen["secondary_generation_sha256"]:
        raise ValueError("secondary training changed after candidate freeze")
    secondary_report = json.loads((root / "secondary-generation-report.json").read_text())
    validate_secondary(secondary_raw, secondary_report, [{"text_id": identity(r["text"])} for r in records] + supplement_views)
    generated = json.loads((root / "audit-generation-report.json").read_text())
    if (generated["output_sha256"] != sha((root / "audit-generation.jsonl").read_bytes()) or
            generated["source_manifest_sha256"] != sha(source_raw) or
            generated["commit"] != os.environ["GITHUB_SHA"] or generated["run_id"] != int(os.environ["GITHUB_RUN_ID"])):
        raise ValueError("audit generation provenance hash mismatch")
    splits = {s: [pair(r) for r in records if r["split"] == s] for s in COUNTS}
    check_splits(splits)
    baseline = json.loads((root / "frozen-baselines.json").read_text())
    formats = {style: [pair(r, style) for r in records if r["split"] == "test"] for style in FORMATS}
    shortcuts = {style: baseline_report(rows, baseline) for style, rows in formats.items()}
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(root / "foundation/tokenizer.json"))
    tok.no_padding()
    tok.no_truncation()
    tokens = {style: [tok.encode(render(r["prompt"], r[side])).ids for r in rows for side in ("chosen", "rejected")]
              for style, rows in formats.items()}
    if any(len(ids) > MAX_TOKENS for batch in tokens.values() for ids in batch):
        raise ValueError("audit full context exceeds package cap")
    wire_identity = {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
                "adapter_hash": frozen["package_keccak"].removeprefix("0x"),
                "tokenizer_hash": FOUNDATION["tokenizer.json"]}
    all_bits, trained, repeats, controls = {}, {}, False, []
    with executor(root, package) as url:
        for style, vectors in tokens.items():
            bits = []
            for start in range(0, len(vectors), 8):
                batch = vectors[start:start + 8]
                bits.extend(check_reply(request(url, batch, wire_identity), len(batch), wire_identity))
            all_bits[style] = bits
            trained[style] = measures(formats[style], bits)
        vectors = tokens["json"][:8]
        repeats = check_reply(request(url, vectors, wire_identity), 8, wire_identity) == all_bits["json"][:8]
        if not repeats:
            raise ValueError("repeat reference score bits differ")
        # The upstream executor returns its actual identity, not an echo. The
        # literal counterparty verifier must reject each wrong request identity.
        for field in wire_identity:
            wrong = {**wire_identity, field: "f" * 64}
            reply = request(url, vectors[:1], wrong)
            try:
                check_reply(reply, 1, wrong)
            except ValueError:
                controls.append(field)
            else:
                raise ValueError("negative identity control accepted")
        prefix_wrong = {**wire_identity, "adapter_hash": "0x" + wire_identity["adapter_hash"]}
        try:
            check_reply(request(url, vectors[:1], prefix_wrong), 1, prefix_wrong)
        except ValueError:
            controls.append("prefixed_adapter_hash")
        else:
            raise ValueError("counterparty bare hash contract not enforced")
        malformed = request(url, [[], [-1], [50368]], wire_identity)
        if any(not item.get("error") or "score" in item for item in malformed.get("results", [])) or len(malformed.get("results", [])) != 3:
            raise ValueError("actual engine invalid-token refusal failed")
    # Fresh process must reproduce the same frozen vectors.
    with executor(root, package) as url:
        if check_reply(request(url, vectors, wire_identity), 8, wire_identity) != all_bits["json"][:8]:
            raise ValueError("fresh-process reference bits differ")
    # Measure seed-matched untrained control in the identical reference engine.
    pins = json.loads((HERE / "model-pins.json").read_text())
    kit = root / ("telegraph-wasm-baseline-" + pins["kit"]["revision"])
    # The bare package hash comes from the primary Go crypto implementation,
    # not a SHA3-256 substitute or an inferred native model identity.
    digest_tool = root / "packagehash"
    subprocess.run(["go", "build", "-trimpath", "-o", str(digest_tool), str(HERE / "packagehash.go")],
                   cwd=kit, timeout=600, check=True)
    untrained_package = root / "UNTRAINED.safetensors"
    untrained_hash = subprocess.run([str(digest_tool), str(untrained_package)], capture_output=True,
                                   text=True, check=True, timeout=20).stdout.strip()
    initial_identity = {**wire_identity, "adapter_hash": untrained_hash}
    untrained = {}
    with executor(root, untrained_package) as url:
        for style, batch in tokens.items():
            bits = []
            for start in range(0, len(batch), 8):
                ids = batch[start:start + 8]
                bits.extend(check_reply(request(url, ids, initial_identity), len(ids), initial_identity))
            untrained[style] = measures(formats[style], bits)
    result = {"family": FAMILY, "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
              "formats": trained, "untrained": untrained, "shortcuts": shortcuts,
              "scope": SCOPE, "audit_used_for_selection": False, "package_sha256": frozen["package_sha256"],
               "source_manifest_sha256": bundle_sha, "normalized_overlap": 0,
              "repeat_bits_identical": repeats, "negative_identity_controls": controls,
              "fixed_vectors": [{"tokens": ids, "score_bits": bits} for ids, bits in zip(vectors, all_bits["json"][:8])],
              "untrained_package_sha256": sha(untrained_package.read_bytes()),
              "untrained_package_keccak": untrained_hash, "audit_pairs": 2 * COUNTS["test"]}
    result.update(quality(trained, untrained, shortcuts))
    dump(root / "audit-runtime-result.json", result)
    if result["verdict"] != "PASS":
        raise ValueError("real frozen audit failed quality gates; preserve and improve with a clean audit")
    # Only observed passes produce a registration envelope.
    ready = {"schema_version": 1, "status": "REGISTRATION_READY", "family": FAMILY,
             "run_id": result["run_id"], "commit": result["commit"],
              "package_sha256": frozen["package_sha256"], "package_keccak": wire_identity["adapter_hash"],
             "foundation_sha256": FOUNDATION["foundation_lora_fp32.onnx"], "engine_sha256": FOUNDATION["foundation_engine.wasm"],
             "tokenizer_sha256": FOUNDATION["tokenizer.json"],
             "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True,
                         "source_manifest_sha256": bundle_sha, "independent_groups": COUNTS["test"]},
             "quality": {"passed": True, "audit_pairs": result["audit_pairs"], "all_supported_formats_tested": True,
                         "formats": {style: {"accuracy": trained[style]["pairwise_accuracy"],
                           "per_label": trained[style]["accuracy_by_label"], "untrained_accuracy": untrained[style]["pairwise_accuracy"],
                           "shortcut_accuracy": max(shortcuts[style].values())} for style in FORMATS}},
             "runtime": {"full_parser_passed": frozen["full_parser_passed"], "reference_engine_passed": True,
                         "repeat_bits_identical": repeats, "negative_identity_controls": len(controls),
                         "vectors": result["fixed_vectors"]}, "serving_commit": result["commit"], "scope": SCOPE}
    dump(root / "registration-ready.json", ready)


if __name__ == "__main__":
    main()
