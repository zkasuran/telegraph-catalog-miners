# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Controller-only bounded CPU training. Importing this file performs no model work."""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import struct
import sys
import time
import zipfile

from build import HERE, sha, verify_splits, write_json
from core import render
import reference

CONTRACT = json.loads((HERE / "contract.json").read_text())
FAMILY = CONTRACT["family"]


def remote_only():
    reference.remote_only()


def shared_modules():
    remote_only()
    licences = {"sand": (HERE / "LICENSE").read_text(), "kit_mit": (HERE / "KIT-LICENSE").read_text(),
                "apache_2_0": (HERE / "UPSTREAM-APACHE-2.0.txt").read_text()}
    licences["notice"] = ("NWS CF6 observations via NOAA/NCEI GHCN-Daily source flag 1, public domain NWS material. "
                          "Source flags and unaltered values retained. Preferences are AI-assisted construction, not human-reviewed. "
                          "ModernBERT by Answer.AI and LightOn, Apache-2.0, revision 8949b909ec900327062f0ebf497f51aef5e6f0c8. "
                          "Telegraph reference kit, copyright 2026 telegraphprotocol, MIT, revision 1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd. "
                          "Our delta, head and code additions, copyright 2026 zkasuran, LicenseRef-zkasuran-SAND-1.0. "
                          "Retained upstream grants apply first to their portions. No endorsement or official climate certification.")
    return reference, licences


def source_kit(root, shared):
    remote_only()
    return shared.source_kit(root)


def stamp_safetensors(raw, licences):
    if not 9 <= len(raw) <= 64 << 20:
        raise ValueError("checkpoint byte ceiling")
    size = struct.unpack("<Q", raw[:8])[0]
    if not 0 < size <= 1 << 20 or size + 8 > len(raw):
        raise ValueError("checkpoint header ceiling")
    header = json.loads(raw[8:8 + size])
    header.setdefault("__metadata__", {}).update({"x-terms-" + k: v for k, v in licences.items()})
    new = json.dumps(header, separators=(",", ":")).encode()
    new += b" " * (-len(new) % 8)
    if len(new) > 1 << 20:
        raise ValueError("stamped header ceiling")
    return struct.pack("<Q", len(new)) + new + raw[8 + size:]


def preserve_checkpoint(destination, licences):
    for filename in destination.glob("*.safetensors"):
        filename.write_bytes(stamp_safetensors(filename.read_bytes(), licences))
    head = destination / "reward_head.pt"
    if head.exists():
        notice = json.dumps(licences, sort_keys=True).encode()
        if len(notice) > 65535:
            raise ValueError("licence ZIP comment ceiling")
        with zipfile.ZipFile(head, "a") as archive:
            archive.comment = notice
        with zipfile.ZipFile(head) as archive:
            if archive.comment != notice or archive.testzip() is not None:
                raise ValueError("reward head ZIP preservation failed")


def read_bound_splits():
    frozen = json.loads((HERE / "freeze.json").read_text())
    for path, expected in frozen["files"].items():
        if sha((HERE / path).read_bytes()) != expected:
            raise ValueError("immutable recipe/data binding mismatch: " + path)
    report = json.loads((HERE / "data/data-report.json").read_text())
    rows = {}
    for split, binding in report["splits"].items():
        raw = (HERE / "data" / (split + ".jsonl")).read_bytes()
        if sha(raw) != binding["sha256"]:
            raise ValueError("frozen split hash mismatch")
        rows[split] = [json.loads(line) for line in raw.splitlines()]
    verify_splits(rows)
    controls = json.loads((HERE / "data/frozen-controls.json").read_text())
    if controls["fit_split"] != "train" or any(type(x) is not int or x not in (-1, 1) for x in controls["integer_coefficients"].values()):
        raise ValueError("invalid frozen train-only shortcut coefficients")
    return report, rows, controls


def model_metadata():
    return {"family": FAMILY, "owner_attempt": CONTRACT["run_id"], "run_id": int(os.environ["GITHUB_RUN_ID"]),
            "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"], "commit": os.environ["GITHUB_SHA"],
            "repository": os.environ["GITHUB_REPOSITORY"], "runtime_identity": CONTRACT["foundation"],
            "contract_sha256": sha((HERE / "contract.json").read_bytes()),
            "node_identity": "pinned reference adapterexec on GitHub runner; not the live node",
            "python_version": sys.version,
            "node_qualification": "not attempted"}


def train(root):
    remote_only()
    preflight = json.loads((HERE / "live-preflight.json").read_text())
    if preflight["class"] != "judgment" or preflight["schema_matches_frozen_task"] is not True:
        raise ValueError("fresh live class/schema gate failed")
    shared, licences = shared_modules()
    report, splits, controls = read_bound_splits()
    kit = source_kit(root, shared)
    sys.path.insert(0, str(kit))
    from float_runtime import train_adapter as trainer
    hp = trainer.Hyperparams(**CONTRACT["hyperparameters"])
    gate = trainer.gate_dataset(splits["train"], hp)
    write_json(root / "dataset-gate.json", gate)
    if not gate["ok"]:
        write_json(root / "result.json", {**model_metadata(), "phase": "dataset", "verdict": "FAIL", "gate": gate})
        raise ValueError("upstream dataset gate failed; no bypass")
    if any(v["accuracy"] > 0.65 for v in controls["validation"].values()):
        raise ValueError("frozen validation shortcut exceeds 65 percent")
    foundation, base, kit = shared.prepare(root)
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, AutoConfig
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import metadata, target_dims, tensor_names, write_package
    if torch.cuda.is_available():
        raise RuntimeError("fixed public CPU workflow refuses GPU")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if not trainer.TOKENIZER_VECTORS.is_file():
        raise ValueError("missing primary tokenizer vectors")
    trainer.check_tokenizer(tok)
    node = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    node.no_padding()
    node.no_truncation()
    checked = 0
    for split in ("train", "validation"):
        for row in splits[split]:
            for side in ("chosen", "rejected"):
                text = render(row["prompt"], row[side])
                if text != trainer.render(row["prompt"], row[side]):
                    raise ValueError("literal trainer render mismatch")
                ids = tok(text, truncation=False)["input_ids"]
                if ids != node.encode(text).ids or len(ids) > hp.max_length:
                    raise ValueError("token mismatch or decisive truncation")
                checked += 1
    write_json(root / "tokenizer-check.json", {"texts": checked, "matches": True, "hash": CONTRACT["foundation"]["tokenizer.json"]})
    cfg = AutoConfig.from_pretrained(base, local_files_only=True)
    foundation_spec = {"foundation_hash": CONTRACT["foundation"]["foundation_lora_fp32.onnx"],
                       "foundation_version": "modernbert-base-lora16-v1", "tokenizer_hash": CONTRACT["foundation"]["tokenizer.json"],
                       "r_max": 16, "num_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
                       "targets": {k: list(v) for k, v in target_dims(cfg).items()}}
    write_json(root / "foundation.json", foundation_spec)
    reference.compile_tools(root, kit)
    def export_package(directory, destination, observed_validation=None):
        tensors, params = read_peft(directory, cfg)
        child = {"intent": FAMILY + "_GENERAL", "dataset_version": sha((HERE / "data/data-report.json").read_bytes()),
                 "pairs": 0 if observed_validation is None else len(splits["validation"]), "held_out": False}
        if observed_validation is not None:
            child["accuracy"] = observed_validation["pairwise_accuracy"]
        training = {"family": FAMILY, "children": [child], "trainer": "fixed CPU preference recipe; development validation selects; fresh independent audit still required"}
        meta = metadata(cls="judgment", intent=FAMILY, foundation_hash=foundation_spec["foundation_hash"],
                        foundation_version=foundation_spec["foundation_version"], rank=params["rank"], alpha=params["alpha"],
                        target_modules=params["target_modules"], max_tokens=hp.max_length, training=training)
        meta.update({"x-terms-" + k: v for k, v in licences.items()})
        meta["x-training-scope"] = CONTRACT["scope"]
        meta["x-metric-scope"] = "unmeasured initialization" if observed_validation is None else "validation, not independent audit"
        write_package(destination, meta, {n: tensors[n] for n in tensor_names(cfg.num_hidden_layers)})
    # Stamp every actual saved checkpoint, including failed runs.
    original_build, original_save_head, original_evaluate = trainer.build_model, trainer.save_reward_head, trainer.evaluate
    def bounded_build(*args):
        model, model_tok = original_build(*args)
        save = model.save_pretrained
        def stamped_save(destination, *save_args, **save_kwargs):
            value = save(destination, *save_args, **save_kwargs)
            preserve_checkpoint(Path(destination), licences)
            return value
        model.save_pretrained = stamped_save
        initial = root / "initial"
        initial.mkdir(exist_ok=True)
        model.save_pretrained(initial)
        original_save_head(model, initial)
        preserve_checkpoint(initial, licences)
        export_package(initial, root / "untrained.safetensors")
        reference.validate_package(root, root / "untrained.safetensors", "untrained-manifest.json")
        return model, model_tok
    def stamped_head(model, destination):
        original_save_head(model, destination)
        preserve_checkpoint(Path(destination), licences)
    def isolated_evaluate(*args, **kwargs):
        py, numeric, tensor = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            return original_evaluate(*args, **kwargs)
        finally:
            random.setstate(py)
            np.random.set_state(numeric)
            torch.set_rng_state(tensor)
    trainer.build_model, trainer.save_reward_head, trainer.evaluate = bounded_build, stamped_head, isolated_evaluate
    def timeout(_signum, _frame):
        raise TimeoutError("fixed 120 minute training ceiling")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(CONTRACT["training_minutes"] * 60)
    started = time.monotonic()
    adapter = root / "adapter"
    model, model_tok, _ = trainer.train(str(base), splits["train"], splits["validation"], adapter, hp)
    signal.alarm(0)
    preserve_checkpoint(adapter, licences)
    (adapter / "tokenizer.json").write_bytes((foundation / "tokenizer.json").read_bytes())
    validation = trainer.evaluate(model, model_tok, splits["validation"], hp)
    package = root / (FAMILY + ".safetensors")
    export_package(adapter, package, validation)
    reference.validate_package(root, package, "package-manifest.json")
    write_json(root / "result.json", {**model_metadata(), "phase": "exported", "verdict": "NEEDS_INDEPENDENT_AUDIT",
               "class_preflight": preflight,
               "package_sha256": sha(package.read_bytes()), "data": report, "gate": gate,
               "untrained_package_sha256": sha((root / "untrained.safetensors").read_bytes()), "validation": validation,
               "elapsed_training_seconds": time.monotonic() - started,
               "versions": {n: importlib.metadata.version(n) for n in ("torch", "numpy", "transformers", "peft", "safetensors", "tokenizers")},
               "node_qualification": "not attempted", "archive_scorer_is_not_neural_model": True})
    del model
    gc.collect()


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    root = parser.parse_args().root
    root.mkdir(parents=True, exist_ok=True)
    train(root)


if __name__ == "__main__":
    main()
