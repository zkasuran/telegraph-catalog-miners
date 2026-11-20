# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""One frozen source-held-out audit in the actual reference engine, no fitting."""
import argparse
import importlib.metadata
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

import contracts
import audit_source
import snapshot
import variant2
from class_binding import collect_classes
from remote import prepare, engine_vectors, shortcuts

HERE = Path(__file__).resolve().parent


def make_pairs(meta, selected):
    if len(selected) < 32:
        raise ValueError("unchanged independent group minimum failed")
    pairs = []
    for group in selected[:32]:
        observation = sorted(group["rows"], key=lambda row: contracts.sha(row["ip"].encode()))[0]
        seed = {"prompt": json.dumps({"source_updated": meta["updated"], "window_days": meta["window_days"], "observation": observation}),
                "group_id": group["group_id"], "source_ip": observation["ip"], "source_asn": observation["asn"],
                "annotation_provenance": "publisher automated first-party observation", "label_kind": "preference_pair"}
        pairs.extend(variant2.formulate(seed))
    for fmt in contracts.FORMATS:
        if sum(row["format"] == fmt for row in pairs) < 64:
            raise ValueError("unchanged independent pair minimum failed")
    return pairs


def main():
    contracts.remote_only()  # Before network, model, tokenizer or engine operations.
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = json.loads((HERE / "candidate-lock.json").read_text())
    report = json.loads((args.candidate / "candidate-report.json").read_text())
    package = args.candidate / "THREAT_IP_REPUTATION.safetensors"
    for name, expected in ((package.name, lock["package_sha256"]), ("untrained.safetensors", lock["untrained_sha256"]),
                           ("frozen-controls.json", lock["frozen_controls_sha256"]), ("candidate-report.json", lock["candidate_report_sha256"])):
        if contracts.sha((args.candidate / name).read_bytes()) != expected:
            raise ValueError("exact frozen candidate/control/report bytes changed")
    if report["run_id"] != lock["selection_run_id"] or report["commit"] != lock["selection_commit"] or report["frozen_at"] != lock["frozen_at"]:
        raise ValueError("immutable selection identity differs")
    source = root / "source-capture"
    development = contracts.read_development(HERE / "data-v2", "compact-elapsed-v2")
    meta, selected, preflight = audit_source.preflight(lock, source, HERE / "audit-base", development["train"] + development["validation"])
    if preflight["status"] != "AUDIT_SOURCE_READY":
        raise ValueError("source is not eligible for neural audit")
    if (root / "audit-started.json").exists():
        raise ValueError("this source/candidate audit already started; reconcile its actual artifact, never evaluate twice")
    pairs = make_pairs(meta, selected)
    normalized = {contracts.norm(contracts.render(row["prompt"], row[side])) for row in pairs for side in ("chosen", "rejected")}
    existing = {contracts.norm(contracts.render(row["prompt"], row[side])) for rows in development.values() for row in rows for side in ("chosen", "rejected")}
    fields = {contracts.norm(row[key]) for row in pairs for key in ("prompt", "chosen", "rejected")}
    dev_fields = {contracts.norm(row[key]) for rows in development.values() for row in rows for key in ("prompt", "chosen", "rejected")}
    if normalized & existing or fields & dev_fields:
        raise ValueError("normalized audit/development overlap")
    audit_raw = b"".join((json.dumps(row, separators=(",", ":")) + "\n").encode() for row in pairs)
    (root / "audit.jsonl").write_bytes(audit_raw)
    start = {"family": contracts.FAMILY, "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
             "package_sha256": lock["package_sha256"], "audit_sha256": contracts.sha(audit_raw), "audit_used_for_selection": False}
    contracts.write_json(root / "audit-started.json", start)
    foundation, base, kit = prepare(root)
    from transformers import AutoConfig
    from tokenizers import Tokenizer
    from float_runtime.lora_format import target_dims
    cfg = AutoConfig.from_pretrained(base, local_files_only=True)
    tokenizer = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=256)
    for row in pairs:
        for side in ("chosen", "rejected"):
            if len(tokenizer.encode(contracts.render(row["prompt"], row[side])).ids) >= 256:
                raise ValueError("audit rendering reaches truncation ceiling")
    subprocess.run(["go", "build", "-trimpath", "-o", str(root / "adapterexec"), "./cmd/adapterexec"], cwd=kit, check=True, timeout=600)
    live_dir = kit / "cmd/threat-owner-livecheck"
    live_dir.mkdir(exist_ok=True)
    shutil.copyfile(HERE / "livecheck.go", live_dir / "main.go")
    subprocess.run(["go", "build", "-trimpath", "-o", str(root / "livecheck"), "./cmd/threat-owner-livecheck"], cwd=kit, check=True, timeout=600)
    collect_classes(root / "live-class-proof.json")
    contracts.write_json(root / "foundation.json", {"foundation_hash": contracts.FOUNDATION["foundation_lora_fp32.onnx"],
                         "foundation_version": "modernbert-base-lora16-v1", "tokenizer_hash": contracts.FOUNDATION["tokenizer.json"],
                         "base_model": str(base), "r_max": 16, "num_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
                         "targets": {key: list(value) for key, value in target_dims(cfg).items()}})
    proofs = {}
    for label, file in (("trained", package), ("untrained", args.candidate / "untrained.safetensors")):
        parsed = subprocess.run([str(root / "livecheck"), "--package", str(file), "--classes", str(root / "live-class-proof.json"),
                                 "--foundation", str(root / "foundation.json"), "--out", str(root)], capture_output=True, text=True, timeout=60)
        contracts.write_json(root / (label + "-parser.json"), {"exit": parsed.returncode, "stdout": parsed.stdout, "stderr": parsed.stderr})
        if parsed.returncode:
            raise ValueError("actual full parser refused frozen " + label + " package")
        manifest = json.loads(parsed.stdout)
        if manifest["sha256"] != contracts.sha(file.read_bytes()):
            raise ValueError("actual parser package hash differs")
        if label == "trained" and manifest["keccak256"].removeprefix("0x") != lock["package_keccak"]:
            raise ValueError("actual frozen candidate keccak differs")
        proofs[label] = engine_vectors(root, foundation, file, manifest, tokenizer, pairs, full_validation=True)
        proofs[label]["scope"] = "one frozen independent source/entity/time audit, no selection or fitting"
        shutil.copyfile(root / "reference-engine.log", root / (label + "-engine.log"))
        contracts.write_json(root / (label + "-runtime.json"), proofs[label])
    controls = json.loads((args.candidate / "frozen-controls.json").read_text())
    if controls["fit_split"] != "train" or controls["train_sha256"] != lock["train_sha256"]:
        raise ValueError("shortcuts are not immutable train-fitted controls")
    formats = {}
    for fmt in contracts.FORMATS:
        values = proofs["trained"]["validation_formats"][fmt]
        formats[fmt] = {**values, "untrained_accuracy": proofs["untrained"]["validation_formats"][fmt]["accuracy"],
                        "shortcut_accuracy": max(shortcuts([row for row in pairs if row["format"] == fmt], controls).values())}
    passed = all(values["accuracy"] >= 0.8 and min(values["per_label"].values()) >= 0.7
                 and values["accuracy"] >= values["untrained_accuracy"] + 0.05
                 and values["accuracy"] >= values["shortcut_accuracy"] + 0.1 for values in formats.values())
    source_receipt = json.loads((source / "receipt.json").read_text())
    ready = {"schema_version": 1, "status": "REGISTRATION_READY" if passed else "AUDIT_QUALITY_FAILED", **start,
             "family": contracts.FAMILY, "package_keccak": lock["package_keccak"], "package_bytes": lock["package_bytes"],
             "foundation_sha256": contracts.FOUNDATION["foundation_lora_fp32.onnx"], "engine_sha256": contracts.FOUNDATION["foundation_engine.wasm"],
             "tokenizer_sha256": contracts.FOUNDATION["tokenizer.json"], "selection": lock,
             "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": source_receipt["grants_verified"],
                        "source_manifest_sha256": contracts.sha((source / "receipt.json").read_bytes()), "independent_groups": len({row["group_id"] for row in pairs}),
                        "audit_sha256": contracts.sha(audit_raw), "source_commit": source_receipt["source_commit"]},
             "quality": {"passed": passed, "audit_pairs": min(sum(row["format"] == fmt for row in pairs) for fmt in contracts.FORMATS),
                         "all_supported_formats_tested": True, "formats": formats}, "runtime": proofs["trained"],
             "serving_commit": os.environ["GITHUB_SHA"], "runtime_node": socket.gethostname(), "fit_on_audit": False,
             "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy", "safetensors")}}
    contracts.write_json(root / "registration-ready.json", ready)
    shutil.copyfile(package, root / package.name)
    print(json.dumps({"status": ready["status"], "formats": formats, "audit_pairs": ready["quality"]["audit_pairs"]}))
    if not passed:
        raise ValueError("actual independent audit failed unchanged quality gates; never tune this exposed audit")


if __name__ == "__main__":
    main()
