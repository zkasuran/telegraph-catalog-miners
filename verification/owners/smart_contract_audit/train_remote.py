# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Train one candidate on train only. Independent audit job scores it once."""
import argparse
import gc
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from data import FAMILY, ATTEMPT, answers, render, sha, write
from policy import execution_gate, remote_only
from remote_prepare import HERE, bound_data, prepare
from contracts import training_pairs


def main():
    remote_only()  # Before model libraries, downloads or model construction.
    owner_attempt = os.environ["OWNER_RUN_ID"]
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    root = p.parse_args().root
    rows, binding = bound_data(root)
    execution_gate(rows, json.loads((root / "evm-report.json").read_text()), binding["inventory_sha256"])
    foundation, base, kit, hashes, sources = prepare(root)
    (kit / "cmd" / "adapterexec" / "owner_wire_contract_test.go").write_bytes((HERE / "wire_contract_test.go").read_bytes())
    checked_wire = subprocess.run(["go", "test", "./cmd/adapterexec", "-run", "^TestOwnerLiteralWire$", "-v"],
                                  cwd=kit, check=True, text=True, capture_output=True, timeout=600)
    (root / "wire-test.log").write_text(checked_wire.stdout + checked_wire.stderr)
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoConfig, AutoTokenizer
    from float_runtime import train_adapter as trainer
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import metadata, tensor_names, write_package, target_dims

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    hp = trainer.Hyperparams(max_length=384, batch_size=2, grad_accum=8, epochs=3,
                            eval_batch=4, gate_min_acc=0.80, grad_checkpointing=False)
    train = training_pairs([r for r in rows if r["split"] == "train"])
    validation = training_pairs([r for r in rows if r["split"] == "validation"])
    gate = trainer.gate_dataset(train, hp)
    write(root / "dataset-gate.json", gate)
    if not gate["ok"]:
        raise ValueError("upstream dataset gate failed, no bypass")
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    node_tok = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    node_tok.no_padding()
    node_tok.enable_truncation(max_length=384)
    maximum, count = 0, 0
    for row in train + validation:
        for style in range(3):
            for answer in answers(row["facts"], row["vulnerable"], style):
                literal = render(row["prompt"], answer)
                if literal != trainer.render(row["prompt"], answer):
                    raise ValueError("literal reference rendering mismatch")
                full = tok(literal, truncation=False)["input_ids"]
                if len(full) > 384:
                    raise ValueError("complete source or answer exceeds scored input")
                if tok(literal, truncation=True, max_length=384)["input_ids"] != node_tok.encode(literal).ids:
                    raise ValueError("tokenizer mismatch")
                maximum = max(maximum, len(full))
                count += 1
    trainer.check_tokenizer(tok)
    write(root / "tokenizer-report.json", {"texts": count, "max_tokens": maximum,
          "cap": 384, "literal_render": True, "tokenizer_sha256": hashes["tokenizer.json"]})
    licences = json.loads((HERE / "licences.json").read_text())
    cfg = AutoConfig.from_pretrained(base, local_files_only=True)
    packages = {}

    def export(directory, name):
        tensors, params = read_peft(directory, cfg)
        # No audit score is invented in the training metadata.
        meta = metadata(cls="judgment", intent=FAMILY, foundation_hash=hashes["foundation_lora_fp32.onnx"],
                        foundation_version="modernbert-base-lora16-v1", rank=params["rank"], alpha=params["alpha"],
                        target_modules=params["target_modules"], max_tokens=384,
                        training={"family": FAMILY, "owner_attempt": owner_attempt,
                                  "data": binding, "audit": "not yet measured",
                                  "children": [{"intent": FAMILY + "_GENERAL", "dataset_version": binding["inventory_sha256"]}],
                                  "authorship": "Original AI-assisted source and constructed preferences; no human review"})
        for key, value in licences.items():
            meta["x-terms-" + key] = value
        package = root / (name + ".safetensors")
        write_package(package, meta, {n: tensors[n] for n in tensor_names(cfg.num_hidden_layers)})
        packages[name] = {"sha256": sha(package.read_bytes()), "metadata": meta}
        return meta

    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    initial, _ = trainer.build_model(str(base), hp)
    initial_dir = root / "untrained"
    initial.save_pretrained(initial_dir)
    trainer.save_reward_head(initial, initial_dir)
    export(initial_dir, "untrained")
    del initial
    gc.collect()
    adapter = root / "adapter"
    try:
        model, _, _ = trainer.train(str(base), train, validation, adapter, hp)
        del model
        gc.collect()
    finally:
        # Preserve the best saved checkpoint under full upstream plus SAND terms,
        # including when a later step fails. This runs no model inference.
        checkpoint = adapter / "adapter_model.safetensors"
        if checkpoint.exists():
            from byte_stamp import stamp
            (root / "checkpoint.safetensors").write_bytes(stamp(checkpoint.read_bytes(), licences))
            (root / "checkpoint-config.json").write_bytes((adapter / "adapter_config.json").read_bytes())
    # Export saved best bytes, never the possibly later in-memory reward head.
    (adapter / "tokenizer.json").write_bytes((foundation / "tokenizer.json").read_bytes())
    meta = export(adapter, "candidate")
    executable, checker = root / "adapterexec", root / "adapterbuild"
    for name, output in (("adapterexec", executable), ("adapterbuild", checker)):
        subprocess.run(["go", "build", "-trimpath", "-o", str(output), "./cmd/" + name], cwd=kit, check=True, timeout=600)
    wire_dir = kit / "cmd" / "owner-wire"
    wire_dir.mkdir(exist_ok=True)
    (wire_dir / "main.go").write_bytes((HERE / "wire_root.go").read_bytes())
    subprocess.run(["go", "build", "-trimpath", "-o", str(root / "wire-root"), "./cmd/owner-wire"], cwd=kit, check=True, timeout=600)
    foundation_spec = {"foundation_hash": meta["foundation_hash"], "foundation_version": meta["foundation_version"],
                      "tokenizer_hash": meta["tokenizer_hash"], "base_model": str(base), "r_max": 16,
                      "num_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
                      "targets": {k: list(v) for k, v in target_dims(cfg).items()}}
    write(root / "foundation.json", foundation_spec)
    for name in packages:
        parsed = subprocess.run([str(checker), "lora", "--intent", FAMILY, "--package", str(root / (name + ".safetensors")),
                                 "--class", "judgment", "--foundation", str(root / "foundation.json"), "--out", str(root / name)],
                                capture_output=True, text=True, timeout=60)
        (root / (name + "-parser.stdout.json")).write_text(parsed.stdout)
        (root / (name + "-parser.stderr.log")).write_text(parsed.stderr)
        parsed.check_returncode()
        manifest = json.loads(parsed.stdout)
        if manifest["sha256"] != packages[name]["sha256"]:
            raise ValueError("counterparty parser hash mismatch")
        packages[name]["manifest"] = manifest
    write(root / "candidate-manifest.json", {"schema_version": 1, "family": FAMILY, "owner_attempt": owner_attempt,
          "run_id": os.environ["GITHUB_RUN_ID"], "commit": os.environ["GITHUB_SHA"], "packages": packages,
          "data": binding, "hyperparameters": asdict(hp), "foundation": hashes,
          "executor_sha256": sha(executable.read_bytes()), "kit": sources["kit"],
          "wire_root_sha256": sha((root / "wire-root").read_bytes()),
          "runtime": {"python": sys.version, "torch": torch.__version__},
          "source_manifest_sha256": sha((HERE / "data-bindings.json").read_bytes()),
          "full_parser_passed": True,
          "phase": "candidate frozen before independent audit", "node_qualification": "not attempted"})
    print("Candidate and seed-matched untrained packages frozen. Audit has not run.")


if __name__ == "__main__":
    main()
