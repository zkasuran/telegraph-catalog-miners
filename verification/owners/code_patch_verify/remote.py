# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Controller-only CPU train/export. Native audit never claims runtime qualification."""
import argparse
import gc
import importlib.metadata
import json
import math
import os
import random
import signal
import struct
import sys
import time
import zipfile
from pathlib import Path

from ci_reference import render
from data import HERE, check_splits, control_metrics, digest, write_json
from static_check import validate_recipe


def remote_only():
    if (os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_REPOSITORY") != "zkasuran/telegraph-catalog-miners"
            or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
            or not os.environ.get("RUNNER_TEMP")):
        raise RuntimeError("neural work is controller-owned public GitHub CPU only")


def caller_contract(path):
    from caller_gate import validate_caller_contract
    return validate_caller_contract(path)


def licence_metadata(shared):
    source = json.loads((shared / "licences.json").read_text())
    return {"x-upstream-apache-2.0": source["apache_2_0"], "x-upstream-mit": source["kit_mit"],
            "x-upstream-notice": (HERE / "NOTICE").read_text(), "x-own-sand": source["sand"],
            "x-licence": "LicenseRef-zkasuran-SAND-1.0"}


def stamp_checkpoint(path, terms):
    if path.stat().st_size > 64 << 20:
        raise ValueError("checkpoint byte ceiling")
    if path.suffix == ".safetensors":
        raw = path.read_bytes()
        size = struct.unpack("<Q", raw[:8])[0]
        if not 0 < size < 1 << 20 or 8 + size > len(raw):
            raise ValueError("checkpoint header ceiling")
        header = json.loads(raw[8:8 + size])
        header["__metadata__"] = {**header.get("__metadata__", {}), **terms}
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + size:])
    elif path.suffix == ".pt":
        with zipfile.ZipFile(path) as archive:
            prefix = [n[:-len("data.pkl")] for n in archive.namelist() if n.endswith("/data.pkl")]
            if len(prefix) != 1:
                raise ValueError("invalid Torch checkpoint ZIP")
            member = prefix[0] + "cpv-licences.json"
            if member in archive.namelist():
                if json.loads(archive.read(member)) != terms:
                    raise ValueError("changed checkpoint terms")
                return
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr(member, json.dumps(terms))


def experiment(root, base, kit, shared, attestation):
    remote_only()
    recipe = json.loads((HERE / "recipe.json").read_text())
    validate_recipe(recipe)
    caller = caller_contract(attestation)
    if not root.resolve().is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("output outside runner temporary storage")
    root.mkdir(exist_ok=True)
    report = json.loads((HERE / "data/data-report.json").read_text())
    splits = {}
    for name in ("train", "validation", "audit"):
        raw = (HERE / "data" / (name + ".jsonl")).read_bytes()
        if digest(raw) != report["files"][name]["sha256"]:
            raise ValueError("frozen data hash mismatch")
        splits[name] = [json.loads(line) for line in raw.splitlines()]
    check_splits(splits)
    controls = json.loads((HERE / "data/frozen-controls.json").read_text())
    if digest((HERE / "data/frozen-controls.json").read_bytes()) != report["controls_sha256"]:
        raise ValueError("immutable baseline bytes changed")
    # Verify complete primary trainer code before importing any neural dependency.
    if digest((kit / "float_runtime/train_adapter.py").read_bytes()) != "5da2919c66c2d0840104628ab88e98a9c8e5e578686f71ca9be4dcd91bd18d71":
        raise ValueError("pinned trainer mismatch")
    sources = json.loads((shared / "sources.json").read_text())
    for name, sha in sources["base"]["files"].items():
        if digest((base / name).read_bytes()) != sha:
            raise ValueError("base identity mismatch")
    terms = licence_metadata(shared)
    write_json(root / "licences.json", terms)
    write_json(root / "run-binding.json", {"owner_attempt": recipe["run_id"], "family": recipe["family"],
               "run": os.environ["GITHUB_RUN_ID"], "commit": os.environ["GITHUB_SHA"], "caller": caller,
               "data": report, "recipe_sha256": digest((HERE / "recipe.json").read_bytes()),
               "runtime_status": "not_checked", "node_status": "not_attempted", "base_sources": sources["base"],
               "node_origin": "https://devnode.telegraphprotocol.com", "reference_kit": sources["kit"],
               "runner_os": os.environ.get("RUNNER_OS"), "runner_arch": os.environ.get("RUNNER_ARCH")})
    sys.path.insert(0, str(kit))
    import numpy as np
    import torch
    from float_runtime import train_adapter as trainer
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import metadata, tensor_names, write_package
    from tokenizers import Tokenizer
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoTokenizer
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only recipe refuses GPU")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    hp = trainer.Hyperparams(**recipe["hyperparameters"])
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if not trainer.TOKENIZER_VECTORS.is_file():
        raise ValueError("tokenizer conformance vectors missing")
    trainer.check_tokenizer(tok)
    tokenizer_path = base / "tokenizer.json"
    if digest(tokenizer_path.read_bytes()) != "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456":
        raise ValueError("released tokenizer identity mismatch")
    node = Tokenizer.from_file(str(tokenizer_path))
    node.no_padding()
    node.no_truncation()
    for rows in splits.values():
        for row in rows:
            for side in ("chosen", "rejected"):
                text = render(row["prompt"], row[side])
                ids = node.encode(text).ids
                if len(ids) > hp.max_length or ids != tok(text, truncation=False)["input_ids"]:
                    raise ValueError("literal evidence/answer truncated or tokenizer mismatch")
    gate = trainer.gate_dataset(splits["train"], hp)
    write_json(root / "primary-data-gate.json", gate)
    if not gate["ok"]:
        raise ValueError("primary data quality gate failed")
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    model, _ = trainer.build_model(str(base), hp)
    parameters = [p for p in model.parameters() if p.requires_grad]
    initial_state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(parameters, lr=hp.lr, weight_decay=hp.weight_decay)
    sampler = torch.Generator().manual_seed(hp.seed)
    loader = DataLoader(splits["train"], batch_size=1, shuffle=True, generator=sampler,
                        collate_fn=trainer.PairCollator(tok, hp.max_length), num_workers=0)

    def evaluate(rows):
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        training = model.training
        with torch.random.fork_rng(devices=[]):
            try:
                return trainer.evaluate(model, tok, rows, hp)
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
                model.train(training)

    def save(destination):
        destination.mkdir(exist_ok=True)
        model.save_pretrained(destination)
        trainer.save_reward_head(model, destination)
        for path in list(destination.glob("*.safetensors")) + list(destination.glob("*.pt")):
            stamp_checkpoint(path, terms)
        tok.save_pretrained(destination)

    best_loss, best_state, selected_epoch = math.inf, None, None
    started = time.monotonic()
    for epoch in range(1, hp.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            rc = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rr = trainer._reward(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
            loss = trainer.bt_loss(rc, rr, batch["margin"])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            (loss / hp.grad_accum).backward()
            if (i + 1) % hp.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                save(root / "checkpoint")
                write_json(root / "checkpoint.json", {"phase": "training", "epoch": epoch, "pair": i + 1,
                           "elapsed_seconds": time.monotonic() - started, "audit_scored": False})
        development = evaluate(splits["validation"])
        if development["val_loss"] < best_loss:
            best_loss = development["val_loss"]
            best_state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            selected_epoch = epoch
        write_json(root / "selection.json", {"selected_epoch": selected_epoch, "validation_loss": best_loss,
                   "selection_split": "validation", "epochs_completed": epoch, "audit_scored": False})
    if best_state is None:
        raise ValueError("no development-selected state")
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(best_state[n])
    save(root / "adapter")
    # Score each audit candidate exactly once after selecting the fixed state.
    model.eval()
    differences = []
    with torch.no_grad():
        for row in splits["audit"]:
            values = []
            for side in ("chosen", "rejected"):
                encoded = tok(render(row["prompt"], row[side]), truncation=False, return_tensors="pt")
                values.append(trainer._reward(model, encoded["input_ids"], encoded["attention_mask"]).item())
            if not all(math.isfinite(value) for value in values):
                raise ValueError("nonfinite audit reward")
            differences.append(values[0] - values[1])
    audit = {"n_eval": len(differences), "pairwise_accuracy": sum(d > 0 for d in differences) / len(differences)}
    ablated = evaluate([{**r, "prompt": ""} for r in splits["audit"]])
    subsets = {}
    for state in (True, False):
        selected = [difference for row, difference in zip(splits["audit"], differences)
                    if json.loads(row["chosen"])["passed"] is state]
        subsets[state] = {"n_eval": len(selected), "pairwise_accuracy": sum(d > 0 for d in selected) / len(selected)}
    cfg = AutoConfig.from_pretrained(base, local_files_only=True)
    tensors, params = read_peft(root / "adapter", cfg)
    meta = metadata(cls="judgment", intent="CODE_PATCH_VERIFY", foundation_hash=
        "0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6",
        foundation_version="modernbert-base-lora16-v1", rank=params["rank"], alpha=params["alpha"],
        target_modules=params["target_modules"], max_tokens=hp.max_length,
        training={"family": "CODE_PATCH_VERIFY", "preference_authorship": "original constructed AI-assisted fixtures",
                  "caller_contract_sha256": digest(attestation.read_bytes()), "selected_epoch": selected_epoch})
    meta.update(terms)
    package = root / "CODE_PATCH_VERIFY.safetensors"
    write_package(package, meta, {n: tensors[n] for n in tensor_names(cfg.num_hidden_layers)})
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(initial_state[n])
    untrained = evaluate(splits["audit"])
    passed = (audit["pairwise_accuracy"] >= 0.80 and audit["pairwise_accuracy"] >= untrained["pairwise_accuracy"] + 0.05
              and ablated["pairwise_accuracy"] <= 0.65 and all(v["pairwise_accuracy"] >= 0.70 for v in subsets.values()))
    write_json(root / "result.json", {"status": "NATIVE_PASS_RUNTIME_PENDING" if passed else "NATIVE_FAIL",
        "audit": audit, "untrained": untrained, "evidence_ablated": ablated,
        "frozen_controls": control_metrics(splits["audit"], controls),
        "subsets": {str(k): v for k, v in subsets.items()}, "package_sha256": digest(package.read_bytes()),
        "runtime_verified": False, "node_qualification": "not_attempted",
        "run_id": os.environ["GITHUB_RUN_ID"], "commit": os.environ["GITHUB_SHA"], "owner_attempt": recipe["run_id"],
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "tokenizers", "numpy")},
        "native_audit_differences": differences, "audit_candidate_evaluations": 2 * len(differences),
        "audit_sha256": report["files"]["audit"]["sha256"], "selected_epoch": selected_epoch})
    gc.collect()
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    for name in ("root", "base", "kit", "shared", "attestation"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.check_only:
        try:
            remote_only()
        except RuntimeError:
            print("PASS: local neural operation refused before model imports")
        else:
            print("Remote context detected. No model operation in check-only mode.")
        try:
            caller_contract(HERE / "caller-contract.json")
        except ValueError:
            print("PASS: missing trusted caller evidence blocks neural work")
        return 0
    remote_only()
    if args.preflight:
        caller_contract(args.attestation or HERE / "caller-contract.json")
        print("Controller caller contract passed structural checks. No model loaded.")
        return 0
    if any(getattr(args, name) is None for name in ("root", "base", "kit", "shared", "attestation")):
        raise ValueError("controller must provide pinned base, kit, shared sources and attestation")
    def stop(_signum, _frame):
        raise TimeoutError("180-minute CPU ceiling")
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(180 * 60)
    return experiment(args.root, args.base, args.kit, args.shared, args.attestation)


if __name__ == "__main__":
    sys.exit(main())
