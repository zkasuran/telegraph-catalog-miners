# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Exact new licensed A1 package, full parser and original reference foundation."""
import json
import subprocess

from contracts import FAMILY, CHILD, FOUNDATION, HERE, remote_only, sha, write
from remote import capture_class


def export(root, name, record):
    remote_only()
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import metadata, target_dims, tensor_names, write_package
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(root / "base", local_files_only=True)
    directory = root / "adapters" / record["snapshot"]
    if sha((directory / "adapter_model.safetensors").read_bytes()) != record["adapter_sha256"] or sha((directory / "adapter_config.json").read_bytes()) != record["config_sha256"]:
        raise ValueError("frozen selected adapter/config bytes differ")
    tensors, parameters = read_peft(directory, cfg)
    recipe = json.loads((HERE / "aegis-plan.json").read_bytes())
    training = {"family": FAMILY, "children": [{"intent": CHILD, "dataset_version": sha((root / "evidence/source-manifest.json").read_bytes()),
                "pairs": record["formats"]["json"]["n"], "accuracy": record["formats"]["json"]["accuracy"], "held_out": False}],
                "selection_scope": "Validation only; the independent human-source audit is measured separately in the actual reference engine.",
                "source_scope": recipe["scope"], "preference_authorship": "Programmatically constructed from original human source labels, not human-reviewed preferences"}
    meta = metadata(cls="judgment", intent=FAMILY, foundation_hash=FOUNDATION["foundation_lora_fp32.onnx"], foundation_version="modernbert-base-lora16-v1",
                    rank=parameters["rank"], alpha=parameters["alpha"], target_modules=parameters["target_modules"], max_tokens=512, training=training)
    licences = json.loads((HERE / "support/training/licences.json").read_bytes())
    for key, source in (("x-upstream-notice", "notice"), ("x-upstream-apache-2.0", "apache_2_0"), ("x-upstream-mit", "kit_mit"), ("x-licence", "sand")):
        meta[key] = licences[source]
    meta["x-nvidia-attribution"] = "NVIDIA Nemotron Content Safety Dataset V2 (Aegis 2.0), Ghosh et al. 2025. Original source text/annotations retain CC BY 4.0. https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
    meta["x-nvidia-cc-by-4.0"] = (HERE / "CC-BY-4.0.txt").read_text()
    package = root / "evidence" / name
    write_package(package, meta, {n: tensors[n] for n in tensor_names(cfg.num_hidden_layers)})
    spec = {"foundation_hash": meta["foundation_hash"], "foundation_version": meta["foundation_version"], "tokenizer_hash": meta["tokenizer_hash"],
            "base_model": str(root / "base"), "r_max": 16, "num_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
            "targets": {key: list(value) for key, value in target_dims(cfg).items()}}
    write(root / "evidence/foundation.json", spec)
    capture_class(root)
    args = [str(root / "stagecheck"), "--intent", FAMILY, "--child", CHILD, "--expected-class", "judgment", "--package", str(package),
            "--sha256", sha(package.read_bytes()), "--foundation", str(root / "evidence/foundation.json"), "--foundation-sha256", sha((root / "evidence/foundation.json").read_bytes()),
            "--class-snapshot", str(root / "evidence/class-snapshot.json"), "--snapshot-sha256", sha((root / "evidence/class-snapshot.json").read_bytes()), "--out", str(root / "evidence")]
    process = subprocess.run(args, capture_output=True, timeout=60)
    (root / "evidence" / (name + ".parser.stdout")).write_bytes(process.stdout)
    (root / "evidence" / (name + ".parser.stderr")).write_bytes(process.stderr)
    if process.returncode or len(process.stdout) > 1 << 20 or len(process.stderr) > 1 << 20:
        raise ValueError("full current package parser refused A1")
    manifest = json.loads(process.stdout)
    write(root / "evidence" / (name + ".parser.json"), manifest)
    return package, manifest
