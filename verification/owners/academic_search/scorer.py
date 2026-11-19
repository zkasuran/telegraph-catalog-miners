# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Guarded ModernBERT reward scorer experiment. Model operations are remote only.

This entry point measures a source-held-out candidate. It never claims full
reference-engine qualification, publishes a package or registers a model.
"""
import argparse
import gc
import importlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

from data import FORMATS, check_splits, render
from source import HERE, sha, write

FAMILY = "ACADEMIC_SEARCH"
REPO = "zkasuran/telegraph-catalog-miners"
KIT = "1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd"
BASE = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
TOKENIZER_SHA = "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456"


def remote_only():
    if (os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_REPOSITORY") != REPO
            or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch" or os.environ.get("RUNNER_OS") != "Linux"):
        raise RuntimeError("neural operations require the bounded public CPU GitHub workflow")
    if not os.environ.get("GITHUB_RUN_ID", "").isdigit() or len(os.environ.get("GITHUB_SHA", "")) != 40:
        raise RuntimeError("actual GitHub run/commit identity required")


def load_freeze(root):
    manifest = json.loads((root / "freeze.json").read_bytes())
    if manifest["family"] != FAMILY or manifest["audit_used_for_selection"] is not False or manifest["formats"] != list(FORMATS):
        raise ValueError("frozen source contract mismatch")
    splits = {}
    for split, count in (("train", 128), ("validation", 32), ("audit", 64)):
        raw = (root / (split + ".json")).read_bytes()
        if sha(raw) != manifest["splits"][split]["sha256"]:
            raise ValueError("frozen split bytes changed")
        rows = json.loads(raw)
        if len(rows) != count or any(r["title_visibility_status"] != "AI-assisted source review before splitting" for r in rows):
            raise ValueError("reviewed independent original quotas not satisfied")
        splits[split] = rows
    check_splits(splits)
    return manifest, splits


def pairs(rows, format_name):
    return [{"prompt": r["prompt"], "chosen": r["formats"][format_name]["chosen"],
             "rejected": r["formats"][format_name]["rejected"], "margin": 0.0} for r in rows]


def training_pairs(rows):
    # One format per original prevents the primary probe's internal split from
    # seeing the same paper in a different wrapper on both sides of its split.
    return [pairs([row], FORMATS[int(row["block_id"], 16) % len(FORMATS)])[0] for row in rows]


def select_epoch(best, epoch, val_loss):
    if type(epoch) is not int or not 1 <= epoch <= 3 or type(val_loss) not in (int, float) or not math.isfinite(val_loss):
        raise ValueError("fixed epoch and finite validation loss required")
    improved = best is None or val_loss < best["val_loss"]
    return ({"epoch": epoch, "val_loss": val_loss} if improved else best), improved


def preserve(model, trainer, destination):
    """Save the trained delta only. Retain full upstream grants before exposure."""
    remote_only()
    import torch
    destination.mkdir(exist_ok=True, parents=True)
    # Full reviewed own and upstream licence metadata is supplied by the public
    # pipeline's byte-preserving checkpoint helper, not a training data grant.
    from checkpoint import licence_metadata, preserve_artifact
    model.save_pretrained(destination)
    trainer.save_reward_head(model, destination)
    before = torch.load(destination / "reward_head.pt", map_location="cpu", weights_only=True)
    for path in sorted(destination.glob("*.safetensors")) + [destination / "reward_head.pt"]:
        preserve_artifact(path)
    after = torch.load(destination / "reward_head.pt", map_location="cpu", weights_only=True)
    if before.keys() != after.keys() or any(not torch.equal(before[k], after[k]) for k in before):
        raise ValueError("licence stamping changed the actual reward head")
    write(destination / "licences.json", licence_metadata())


def fit(frozen, base, kit, tokenizer_path, out):
    remote_only()  # Must precede every neural or training-library import.
    manifest, splits = load_freeze(frozen)
    if sha(tokenizer_path.read_bytes()) != TOKENIZER_SHA:
        raise ValueError("released tokenizer bytes mismatch")
    # The source kit comes from a hash-verified immutable public archive.
    identity = json.loads((kit / "source-identity.json").read_bytes())
    if identity.get("commit") != KIT or identity.get("verified_archive_sha256") != "18d7dbe7425e977b576a70146f901a006b8e4abddbfd351276205faa7f287243":
        raise ValueError("source kit identity mismatch")
    identity = json.loads((base / "source-identity.json").read_bytes())
    if identity.get("revision") != BASE:
        raise ValueError("base source identity mismatch")
    expected_base = {"model.safetensors": "340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9",
                     "config.json": "1609d59e627c33eaed524b4f01e546d42e84190a079a5a5ded84b212c41c324f",
                     "tokenizer_config.json": "3cd2017ff46d0a527e5d39cae39272eccfa1f19bb9f89b05d166aab2e38354e2",
                     "special_tokens_map.json": "ea97ecdbcc73713039d8d64dbb05e3689495c96657fbd9a18f5bed381be81049"}
    for name, expected in expected_base.items():
        if sha((base / name).read_bytes()) != expected:
            raise ValueError("actual base bytes mismatch: " + name)
    if sha((kit / "float_runtime/train_adapter.py").read_bytes()) != "5da2919c66c2d0840104628ab88e98a9c8e5e578686f71ca9be4dcd91bd18d71":
        raise ValueError("actual primary trainer source changed")
    development = json.loads((frozen / "development-controls-v2.json").read_bytes())
    if not development["training_allowed"] or development["audit_scored"] is not False:
        raise ValueError("development/source controls refused training")
    sys.path.insert(0, str(kit))
    trainer = importlib.import_module("float_runtime.train_adapter")
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from torch.utils.data import DataLoader
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only experiment refuses GPU runtime")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    hp = trainer.Hyperparams(max_length=256, batch_size=1, grad_accum=16, eval_batch=1,
                            lora_rank=8, lora_alpha=32, epochs=3, seed=42, grad_checkpointing=False)
    node = Tokenizer.from_file(str(tokenizer_path))
    node.no_padding()
    node.no_truncation()
    train = training_pairs(splits["train"])
    gate = trainer.gate_dataset(train, hp)
    out.mkdir(exist_ok=True, parents=True)
    write(out / "upstream-training-gate.json", gate)
    if not gate["ok"]:
        raise ValueError("unmodified primary dataset gate refused the candidate")
    # Immutable train-fitted baseline coefficients must already be present.
    baseline = json.loads((frozen / "frozen-baselines.json").read_bytes())
    if baseline.get("training_sha256") != manifest["splits"]["train"]["sha256"] or baseline.get("fit_split") != "train":
        raise ValueError("immutable train-fitted baseline identity missing")
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    model, tok = trainer.build_model(str(base), hp)
    trainer.check_tokenizer(tok)
    for rows in splits.values():
        for f in FORMATS:
            for pair in pairs(rows, f):
                for side in ("chosen", "rejected"):
                    text = render(pair["prompt"], pair[side])
                    expected = node.encode(text).ids
                    if len(expected) > 256 or tok(text, truncation=False)["input_ids"] != expected:
                        raise ValueError("full label object does not reach the node tokenizer")
    def state_hash(current):
        digest = __import__("hashlib").sha256()
        for name, p in current.named_parameters():
            if p.requires_grad:
                digest.update(name.encode())
                digest.update(p.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
    initial_hash = state_hash(model)
    preserve(model, trainer, out / "initial")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=hp.lr, weight_decay=hp.weight_decay)
    sampler = torch.Generator(device="cpu").manual_seed(hp.seed)
    loader = DataLoader(train, batch_size=1, shuffle=True, collate_fn=trainer.PairCollator(tok, 256), generator=sampler)
    best, selected, selected_hash, history = None, None, None, []
    start, timings = time.monotonic(), []
    for epoch in range(1, 4):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step = time.monotonic()
        for i, batch in enumerate(loader):
            chosen = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rejected = trainer._reward(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
            loss = trainer.bt_loss(chosen, rejected, batch["margin"])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            (loss / hp.grad_accum).backward()
            if (i + 1) % hp.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                timings.append(time.monotonic() - step)
                preserve(model, trainer, out / "checkpoint")
                # Checkpoint journals are versioned, never rewritten after failure.
                write(out / ("step-" + str(len(timings)) + ".json"), {"epoch": epoch, "actual_steps": len(timings), "seconds": timings[-1], "audit_scored": False})
                remaining = 3 * len(loader) / hp.grad_accum - len(timings)
                if time.monotonic() - start + remaining * max(timings) > 150 * 60:
                    raise TimeoutError("measured CPU throughput exceeds training reserve")
                step = time.monotonic()
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            with torch.random.fork_rng(devices=[]):
                metrics = trainer.evaluate(model, tok, pairs(splits["validation"], "title"), hp)
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
        best, improved = select_epoch(best, epoch, metrics["val_loss"])
        if improved:
            selected = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
            selected_hash = state_hash(model)
            preserve(model, trainer, out / ("selected-epoch-" + str(epoch)))
        history.append({"epoch": epoch, "validation": metrics, "selected": improved, "audit_scored": False})
    if selected is None:
        raise ValueError("validation did not select a candidate")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(selected[name])
    if state_hash(model) != selected_hash:
        raise ValueError("selected reward head or LoRA delta was not restored")
    preserve(model, trainer, out / "selected")
    report = {"schema_version": 1, "status": "FROZEN_MODEL_RUNTIME_PENDING", "family": FAMILY,
              "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"], "foundation_revision": BASE,
              "source_manifest_sha256": manifest["source_manifest_sha256"], "selected": best, "history": history,
              "initial_sha256": initial_hash, "selected_trainable_sha256": selected_hash, "audit_used_for_selection": False,
              "optimizer_steps": len(timings), "step_seconds": timings, "runtime_verified": False,
              "audit_pending": True, "registration_ready": False}
    report["selected_files"] = {p.name: sha(p.read_bytes()) for p in sorted((out / "selected").iterdir()) if p.is_file()}
    report["initial_files"] = {p.name: sha(p.read_bytes()) for p in sorted((out / "initial").iterdir()) if p.is_file()}
    report["frozen_baseline_sha256"] = sha((frozen / "frozen-baselines.json").read_bytes())
    report["unseen_term_baseline_sha256"] = sha((frozen / "unseen-term-baselines.json").read_bytes())
    write(out / "selected-model-report.json", report)
    # Independent audit and reference-engine export run in separate packaged
    # stages after this immutable selected checkpoint. No missing stage is a pass.
    del model, optimizer, parameters, selected, loader
    gc.collect()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("frozen", "base", "kit", "tokenizer", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fit(args.frozen, args.base, args.kit, args.tokenizer, args.out), indent=2))
