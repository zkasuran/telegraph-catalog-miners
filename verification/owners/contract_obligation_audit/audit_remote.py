# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""One independent audit on the frozen packages in the actual reference engine."""
import argparse
import http.client
import json
import os
import subprocess
import threading
from collections import Counter
from pathlib import Path

from baselines import measure, validate_frozen
from contracts import FAMILY, FORMATS, FOUNDATION, HERE, LABELS, check_config, decode_bits, file_sha, quality, read_jsonl, remote_only, render, strict_json, write_json
from data import check_splits, pair
from executor import ReferenceExecutor, Server
from prepare_remote import compile_runtime, full_parser, prepare
from validation_policy import require_validation
from source_cues import measure_source_cues


def request(tokens, package_keccak):
    return {"batch": [{"tokens": row} for row in tokens], "foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
            "adapter_hash": package_keccak, "tokenizer_hash": FOUNDATION["tokenizer.json"]}


def audit(root):
    remote_only()
    config = check_config(strict_json((HERE/"config.json").read_bytes()))
    freeze = strict_json((root/"training-freeze.json").read_bytes())
    require_validation(freeze["validation"], config["limits"]["validation"])
    if freeze["commit"] != os.environ["GITHUB_SHA"] or freeze["run_id"] != int(os.environ["GITHUB_RUN_ID"]) or freeze["audit_used_for_selection"] is not False:
        raise ValueError("training freeze is not from this exact immutable run")
    source_path = root/"data/source-manifest.json"
    source = strict_json(source_path.read_bytes())
    if file_sha(source_path) != freeze["source_manifest_sha256"] or source["independent_groups"] < 32 or source["normalized_overlap"] != 0:
        raise ValueError("source audit boundary or manifest identity mismatch")
    splits = {name: read_jsonl(root/"data"/(name+".jsonl")) for name in config["limits"]}
    for name, rows in splits.items():
        if file_sha(root/"data"/(name+".jsonl")) != source["files"][name]["sha256"]:
            raise ValueError("audit data changed after candidate freeze")
    check_splits(splits, config)
    policy = strict_json((HERE/"source-policy.json").read_bytes())
    retirement = HERE/"retired-source-groups.json"
    if retirement.exists():
        retired = strict_json(retirement.read_bytes())
        if any(row["source_group"] in retired["groups"] for row in splits["audit"]):
            raise ValueError("previously exposed audit group is permanently retired")
    if any(r["source_group"] in policy["audit_forbidden_groups"] for r in splits["audit"]):
        raise ValueError("exposed source entered independent audit")
    frozen = strict_json((HERE/"frozen-baselines-v7.json").read_bytes())
    source_baseline = strict_json((HERE/"frozen-source-cues-v7.json").read_bytes())
    if source_baseline["training_data_sha256"] != source["files"]["train"]["sha256"]:
        raise ValueError("audit semantic baseline is not from the frozen training data")
    validate_frozen(frozen, splits["train"])
    foundation, _, kit = prepare(root)
    compile_runtime(root, kit)
    # These imports and all actual tokenization/inference are remote only.
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(foundation/"tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    views, vectors = {}, {}
    for style in FORMATS:
        views[style] = [pair(row, style) for row in splits["audit"]]
        vectors[style] = []
        for row in views[style]:
            for side in ("chosen", "rejected"):
                ids = tokenizer.encode(render(row["prompt"], row[side])).ids
                if not ids or len(ids) > config["max_tokens"]:
                    raise ValueError("audited differing claim would be truncated")
                vectors[style].append(ids)
    scores, parsers, fixed, negative_controls = {}, {}, [], []
    for name, expected in (("untrained", freeze["untrained_sha256"]), (FAMILY, freeze["candidate_sha256"])):
        package = root/(name+".safetensors")
        if file_sha(package) != expected:
            raise ValueError("frozen package bytes changed")
        manifest = full_parser(root, package, root/"foundation.json")
        parsers[name] = manifest
        package_keccak = manifest["keccak256"].removeprefix("0x")
        backend = ReferenceExecutor(root, package, package_keccak)
        server = worker = None
        try:
            scores[name] = {}
            for style in FORMATS:
                bits = []
                for start in range(0, len(vectors[style]), 8):
                    batch = vectors[style][start:start+8]
                    result = backend.score(request(batch, package_keccak))
                    bits.extend(item["score"] for item in result["results"])
                scores[name][style] = bits
                write_json(root/"audit-score-bits.json", scores)
            if name == FAMILY:
                # Fixed controls are separate from audit selection/accuracy.
                tokens = [[50281]+[22+i]*(length-2)+[50282] for i, length in enumerate((4, 8, 16, 24, 32, 64, 128, 512))]
                first = backend.score(request(tokens, package_keccak))
                repeated = backend.score(request(tokens, package_keccak))
                if first != repeated:
                    raise ValueError("actual reference-engine repeat bits differ")
                fixed = [{"tokens": token, "score_bits": item["score"]} for token, item in zip(tokens, first["results"])]
                server = Server(("127.0.0.1", 0), backend)
                worker = threading.Thread(target=server.serve_forever, daemon=True)
                worker.start()
                def refusal(value, control):
                    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                    try:
                        connection.request("POST", "/runnable", json.dumps(value), {"Content-Type": "application/json"})
                        response = connection.getresponse()
                        raw = response.read((1 << 20)+1)
                        if response.status != 400 or strict_json(raw) != {"error": "request refused"}:
                            raise ValueError("actual literal HTTP refusal control failed")
                        negative_controls.append({"control": control, "actual_http_status": response.status})
                    finally:
                        connection.close()
                for key in ("foundation_hash", "adapter_hash", "tokenizer_hash"):
                    wrong = request(tokens[:1], package_keccak)
                    wrong[key] = "0x"+wrong[key]
                    refusal(wrong, key)
                for invalid in ([], [-1], [50368], [50281]*513):
                    refusal(request([invalid], package_keccak), "invalid_tokens_"+str(len(invalid)))
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()
                worker.join(timeout=10)
            backend.close()
    formats = {}
    for style in FORMATS:
        values = [decode_bits(bit) for bit in scores[FAMILY][style]]
        initial = [decode_bits(bit) for bit in scores["untrained"][style]]
        wins = [values[2*i] > values[2*i+1] for i in range(len(views[style]))]
        before = [initial[2*i] > initial[2*i+1] for i in range(len(views[style]))]
        labels = Counter(row["label"] for row in views[style])
        shortcuts = measure(views[style], frozen)
        shortcuts.update(measure_source_cues(views[style], source_baseline))
        formats[style] = {"accuracy": sum(wins)/len(wins), "untrained_accuracy": sum(before)/len(before),
            "per_label": {label: sum(w for row, w in zip(views[style], wins) if row["label"] == label)/labels[label] for label in LABELS},
            "shortcut_accuracy": max(shortcuts.values()), "shortcuts": shortcuts, "audit_pairs": len(wins)}
    result = quality(formats)
    manifest = parsers[FAMILY]
    report = {"schema_version": 1, "status": "REGISTRATION_READY" if result["passed"] else "QUALITY_FAILED",
        "family": FAMILY, "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
        "package_sha256": manifest["sha256"], "package_keccak": manifest["keccak256"].removeprefix("0x"),
        "package_bytes": manifest["bytes"], "foundation_sha256": FOUNDATION["foundation_lora_fp32.onnx"],
        "engine_sha256": FOUNDATION["foundation_engine.wasm"], "tokenizer_sha256": FOUNDATION["tokenizer.json"],
        "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True,
                   "source_manifest_sha256": file_sha(source_path), "independent_groups": source["independent_groups"]},
        "quality": {**result, "audit_pairs": len(splits["audit"]), "all_supported_formats_tested": True, "formats": formats},
        "runtime": {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
                    "negative_identity_controls": 3, "negative_controls_observed": negative_controls, "vectors": fixed,
                    "executor_sha256": file_sha(root/"adapterexec"), "go_version": subprocess.check_output(["go", "version"], text=True).strip()},
        "serving_commit": os.environ["GITHUB_SHA"], "scope": "Four NDA claim-entailment topics, not performance compliance",
        "training_freeze_sha256": file_sha(root/"training-freeze.json")}
    write_json(root/"registration-ready.json", report)
    print(json.dumps({"status": report["status"], "formats": formats, "reference_vectors": len(fixed)}, indent=2))
    return 0 if result["passed"] else 2


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    return audit(parser.parse_args().root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
