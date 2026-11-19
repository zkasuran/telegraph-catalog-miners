# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Recover the actual failed parser/runtime with identical tensor payloads.

This stage does not retrain, select on audit or authorize registration.
All model execution and checkpoint repacking run only on the owned CPU runner.
"""
import argparse
import copy
import hashlib
import json
import os
import struct
import subprocess
import sys
import tarfile
from pathlib import Path

from contracts import BASE_REVISION, FAMILY, FOUNDATION, KIT, KIT_SHA, VALIDATION_SHA, read_development, remote_only, sha, validation_gate, write_json
from remote import BASE_FILES, RELEASE, engine_vectors, fetch, shortcuts
from training_record import validate_record
from class_binding import collect_classes
import shutil

HERE = Path(__file__).resolve().parent
INPUTS = {"THREAT_IP_REPUTATION.safetensors": "223f3c997d44b4fba51ca07e325b2ef15b0e7b9d520eb318101b311c4cea8994",
          "untrained.safetensors": "8f0fd804f385998d0cc773b23803aa1af392a2932ae4a060f52f49e3efbccc9d",
          "frozen-controls.json": "d7dd49f438240f6d646242d908bcb09fe7fd989d4529c9dccd65733d88ec66b4"}


def rewritten_header(header, trained):
    fixed = copy.deepcopy(header)
    metadata = fixed["__metadata__"]
    original = metadata.pop("training")
    metadata["x-original-training-provenance"] = original
    metadata["x-integration-recovery"] = "TrainingRecord shape corrected; tensor bytes unchanged. Development is not audit. No quality or canonical win claimed."
    if trained:
        # The original real training log reports epoch-0 restored development
        # accuracy 0.5000 on 64 pairs. No per-format or Spearman value is inferred.
        record = {"family": FAMILY, "children": [{"intent": FAMILY + "_GENERAL", "dataset_version": VALIDATION_SHA,
                                                   "pairs": 64, "accuracy": 0.5, "held_out": False}]}
        validate_record(record)
        metadata["training"] = json.dumps(record, separators=(",", ":"))
        metadata["x-record-accuracy-scope"] = "original native aggregate development log, not reference engine or independent audit; Spearman unmeasured and omitted"
    return fixed


def repack(source, target, trained):
    remote_only()
    with source.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < size <= 1 << 20:
            raise ValueError("checkpoint header ceiling")
        header = json.loads(stream.read(size))
        fixed = rewritten_header(header, trained)
        if {name: value for name, value in fixed.items() if name != "__metadata__"} != {name: value for name, value in header.items() if name != "__metadata__"}:
            raise ValueError("recovery changed tensor descriptors")
        raw = json.dumps(fixed, sort_keys=True, separators=(",", ":")).encode()
        raw += b" " * ((8 - len(raw) % 8) % 8)
        digest = hashlib.sha256()
        with target.open("wb") as output:
            output.write(struct.pack("<Q", len(raw)))
            output.write(raw)
            while chunk := stream.read(1 << 20):
                digest.update(chunk)
                output.write(chunk)
    with target.open("rb") as stream:
        stream.seek(8 + len(raw))
        check = hashlib.sha256()
        while chunk := stream.read(1 << 20):
            check.update(chunk)
    if check.digest() != digest.digest():
        raise ValueError("recovery changed actual tensor payload bytes")
    return {"input_sha256": sha(source.read_bytes()), "output_sha256": sha(target.read_bytes()),
            "tensor_payload_sha256": digest.hexdigest(), "tensor_descriptors_identical": True, "tensor_payload_identical": True,
            "original_grants_retained": all(fixed["__metadata__"].get(key) == value for key, value in header["__metadata__"].items() if key.startswith("x-"))}


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name, digest in INPUTS.items():
        if sha((args.input / name).read_bytes()) != digest:
            raise ValueError("recovery inputs differ from the actual failed job")
    data = read_development(HERE / "data")
    foundation = root / "foundation"
    foundation.mkdir(exist_ok=True)
    for name, digest in FOUNDATION.items():
        fetch(RELEASE + name, foundation / name, digest)
    config = root / "config.json"
    fetch("https://huggingface.co/answerdotai/ModernBERT-base/resolve/" + BASE_REVISION + "/config.json", config, BASE_FILES["config.json"], 1 << 20)
    cfg = json.loads(config.read_text())
    archive = root / "kit.tar.gz"
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + KIT, archive, KIT_SHA, 32 << 20)
    with tarfile.open(archive) as source:
        if sum(member.size for member in source.getmembers()) > 128 << 20:
            raise ValueError("reference source ceiling")
        source.extractall(root, filter="data")
    kit = root / ("telegraph-wasm-baseline-" + KIT)
    for name in ("adapterexec", "adapterbuild"):
        subprocess.run(["go", "build", "-trimpath", "-o", str(root / name), "./cmd/" + name], cwd=kit, check=True, timeout=600)
    live_dir = kit / "cmd/threat-owner-livecheck"
    live_dir.mkdir(exist_ok=True)
    shutil.copyfile(HERE / "livecheck.go", live_dir / "main.go")
    subprocess.run(["go", "build", "-trimpath", "-o", str(root / "livecheck"), "./cmd/threat-owner-livecheck"], cwd=kit, check=True, timeout=600)
    h, i = cfg["hidden_size"], cfg["intermediate_size"]
    spec = {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"], "foundation_version": "modernbert-base-lora16-v1",
            "tokenizer_hash": FOUNDATION["tokenizer.json"], "r_max": 16, "num_layers": cfg["num_hidden_layers"], "hidden_size": h,
            "targets": {"attn.Wqkv": [h, 3*h], "attn.Wo": [h, h], "mlp.Wi": [h, 2*i], "mlp.Wo": [i, h]}}
    write_json(root / "foundation.json", spec)
    def parse(package, live=False):
        if live:
            collect_classes(root / "live-class-proof.json")
            command = [str(root / "livecheck"), "--package", str(package), "--classes", str(root / "live-class-proof.json"),
                       "--foundation", str(root / "foundation.json"), "--out", str(root)]
        else:
            command = [str(root / "adapterbuild"), "lora", "--intent", FAMILY, "--package", str(package), "--class", "judgment",
                       "--foundation", str(root / "foundation.json"), "--out", str(root)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        return {"exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    raw_result = parse(args.input / (FAMILY + ".safetensors"))
    write_json(root / "original-parser-result.json", raw_result)
    if raw_result["exit"] == 0 or "compiled-in catalog" not in raw_result["stderr"]:
        raise ValueError("actual original parser failure differs; preserve its exact stderr before another repair")
    live_original = parse(args.input / (FAMILY + ".safetensors"), live=True)
    write_json(root / "live-class-original-parser-result.json", live_original)
    if live_original["exit"] == 0 or "children is empty" not in live_original["stderr"]:
        raise ValueError("live-bound original package failure differs; preserve exact stderr before metadata repair")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=512)
    recovery, runtimes = {}, {}
    for key, name, trained in (("trained", FAMILY + ".safetensors", True), ("untrained", "untrained.safetensors", False)):
        output = root / name
        recovery[key] = repack(args.input / name, output, trained)
        parsed = parse(output, live=True)
        write_json(root / (key + "-parser-result.json"), parsed)
        if parsed["exit"]:
            raise ValueError("corrected checkpoint still fails full parser")
        manifest = json.loads(parsed["stdout"])
        write_json(root / (key + "-package-manifest.json"), manifest)
        runtimes[key] = engine_vectors(root, foundation, output, manifest, tokenizer, data["validation"], full_validation=True)
        (root / (key + "-reference-engine.log")).write_bytes((root / "reference-engine.log").read_bytes())
        write_json(root / (key + "-runtime-proof.json"), runtimes[key])
    controls = json.loads((args.input / "frozen-controls.json").read_text())
    from contracts import FORMATS
    strongest = {fmt: max(shortcuts([row for row in data["validation"] if row["format"] == fmt], controls).values()) for fmt in FORMATS}
    gate = validation_gate(runtimes["trained"]["validation_formats"], runtimes["untrained"]["validation_formats"], strongest)
    report = {"schema_version": 1, "family": FAMILY, "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
              "original_training_run": 44586142914, "original_training_commit": "f2d2c59fc410d23b9434de11341c944328972804",
              "status": "INTEGRATION_RECOVERED_QUALITY_FAILED" if not gate["passed"] else "INTEGRATION_RECOVERED_AUDIT_REQUIRED",
              "retrained": False, "recovery": recovery, "runtimes": runtimes, "validation_gate": gate,
              "audit_pairs": 0, "audit_used_for_selection": False, "qualification_ready": False,
              "limits": ["Reference-engine development metrics, not a fresh independent audit or canonical node result.",
                         "Tensor-preserving metadata integration recovery does not repair a genuine quality loss."]}
    write_json(root / "recovery-report.json", report)
    print(json.dumps({"status": report["status"], "validation_gate": gate, "retrained": False, "audit_pairs": 0}))


if __name__ == "__main__":
    main()
