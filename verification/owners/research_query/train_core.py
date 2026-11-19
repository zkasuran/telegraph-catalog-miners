# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Fixed remote CPU candidate. Nothing neural executes before remote_only."""
import argparse
import gc
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

from common import CHILD, FAMILY, HERE, SHARED, capture_class, compile_checker, module, prepare_foundation, remote_only, source_kit
from prepare import check_splits, render, replay, trainer_module
from scorer import score as symbolic_score
from source import TOKENIZER_SHA, sentence_candidates, sha, write_json
from terms import metadata as licence_metadata, preserve, stamp


def read_data():
    contract = json.loads((HERE / "contract.json").read_text())
    splits = {}
    for name, pin in contract["splits"].items():
        raw = (HERE / "data" / (name + ".jsonl")).read_bytes()
        if len(raw) > 8 << 20 or sha(raw) != pin["sha256"]:
            raise ValueError("frozen split identity mismatch")
        splits[name] = [json.loads(line) for line in raw.splitlines()]
        if len(splits[name]) != pin["rows"] or len({r["group_id"] for r in splits[name]}) != pin["groups"]:
            raise ValueError("frozen row or group count mismatch")
    check_splits(splits)
    return contract, splits


def quality(trained, untrained, controls, ablation):
    values = [trained, untrained, ablation, *controls.values()]
    if not controls or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("invalid quality report")
    failures = []
    if trained < .80:
        failures.append("audit ordering below 80 percent")
    if trained < untrained + .05:
        failures.append("less than five points above untrained initialization")
    if trained < max(controls.values()) + .10 or max(controls.values()) > .65:
        failures.append("shortcut margin or maximum shortcut gate failed")
    if ablation > .65:
        failures.append("evidence-removed audit exceeds 65 percent")
    return {"passed": not failures, "failures": failures}


def runtime_check(root, kit, rows, native):
    remote_only()
    from tokenizers import Tokenizer
    # Reuse only the existing executor transport and identity/finite-bit checks.
    # Load it into an isolated namespace and configure this process's family.
    sys.path.insert(0, str(SHARED / "qa-experiment"))
    contract = json.loads((HERE / "contract.json").read_text())
    for name in ("qa-experiment/data.py", "qa-experiment/live_class.py"):
        if sha((SHARED / name).read_bytes()) != contract["shared_source_sha256"][name]:
            raise ValueError("runtime dependency source changed")
    transport = module("research_shared_runtime", SHARED / "qa-experiment/runtime_check.py")
    sys.path.pop(0)
    transport.FAMILY, transport.CHILD = FAMILY, CHILD
    subprocess.run(["go", "build", "-mod=readonly", "-trimpath", "-o", str(root / "adapterexec"), "./cmd/adapterexec"],
                   cwd=kit, check=True, timeout=600)
    snapshot = capture_class(root)
    package = root / (FAMILY + ".safetensors")
    command = [str(root / "packagecheck"), "--intent", FAMILY, "--child", CHILD,
               "--class-snapshot", str(root / "class-snapshot.json"), "--snapshot-sha256", snapshot,
               "--package", str(package), "--sha256", sha(package.read_bytes()), "--foundation", str(root / "foundation.json"),
               "--foundation-sha256", sha((root / "foundation.json").read_bytes()), "--out", str(root)]
    completed = subprocess.run(command, capture_output=True, check=True, timeout=60)
    manifest = json.loads(completed.stdout)
    write_json(root / "package-manifest.json", manifest)
    tokenizer = Tokenizer.from_file(str(root / "foundation/tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    vectors = [tokenizer.encode(render(r["prompt"], r[s])).ids for r in rows for s in ("chosen", "rejected")]
    if len(vectors) != 128 or any(len(ids) > 256 for ids in vectors):
        raise ValueError("complete runtime input count/cap mismatch")
    with transport.server(root, manifest, "first") as executor:
        streams = [[b for start in range(0, len(vectors), 4) for b in executor(vectors[start:start + 4])] for _ in range(2)]
        refused = 0
        for key in ("foundation_hash", "tokenizer_hash", "adapter_hash"):
            try:
                executor(vectors[:1], {key: "0" * 64})
            except ValueError:
                refused += 1
        if refused != 3:
            raise ValueError("wrong runtime identities escaped verification")
    with transport.server(root, manifest, "fresh") as executor:
        fresh = [b for start in range(0, 8, 4) for b in executor(vectors[start:start + 4])]
    if streams[0] != streams[1] or fresh != streams[0][:8]:
        raise ValueError("repeated or fresh-process FP32 bits changed")
    agreement = transport.native_agreement(native, streams[0])
    decoded = [transport.decode_bits(b) for b in streams[0]]
    accuracy = sum(decoded[i] > decoded[i + 1] for i in range(0, 128, 2)) / 64
    write_json(root / "runtime.json", {"repeated_bits": streams, "fresh_bits": fresh, "native_agreement": agreement,
               "accuracy": accuracy, "pairs": 64, "independent_groups": 32, "wrong_identity_controls": refused,
               "vectors_sha256": sha(json.dumps(vectors).encode()), "node": "not qualified",
               "run_id": os.environ["GITHUB_RUN_ID"], "commit": os.environ["GITHUB_SHA"], "package": manifest})
    return agreement["numerical"] and agreement["ordering"] and accuracy >= .8


def experiment(root):
    remote_only()
    started = time.monotonic()
    contract, splits = read_data()
    preserve(root)
    kit = source_kit(root)
    trainer = trainer_module(kit / "float_runtime/train_adapter.py")
    hp = trainer.Hyperparams(max_length=256, lora_rank=8, lora_alpha=32, batch_size=1, grad_accum=16,
                            epochs=3, eval_batch=1, early_stopping=False, grad_checkpointing=False, gate_min_acc=.8)
    upstream = trainer.gate_dataset(splits["train"], hp)
    if not upstream["ok"] or upstream["metrics"].get("surface_probe_acc") is None:
        write_json(root / "dataset-gate.json", upstream)
        raise ValueError("unchanged upstream dataset gate failed")
    compile_checker(root, kit, "go")
    # Re-download only the 112 selected pinned originals and independently
    # bind each complete sentence before any model operations.
    fetcher = module("research_original_fetch", SHARED / "training/data.py")
    originals = {r["source_doi"]: r for rows in splits.values() for r in rows}
    source_root = root / "source-originals"
    source_root.mkdir()
    for row in originals.values():
        path = source_root / row["source_path"]
        fetcher.fetch(row["source_url"], path, row["source_sha256"], 4 << 20)
        candidates, _ = sentence_candidates(path.read_bytes(), row["source_path"], row["git_blob_sha1"])
        if not any(c["evidence"] == row["evidence"] and c["license_uri"] == row["license_uri"] for c in candidates):
            raise ValueError("original excerpt or grant binding mismatch")
    write_json(root / "source-check.json", {"grants_verified": True, "originals_reverified": len(originals), "normalized_overlap": source_observed, "independent_groups": len({r["group_id"] for r in splits["audit"]}), "audit_used_for_selection": False})
    foundation, base, kit = prepare_foundation(root)
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from torch.utils.data import DataLoader
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only job refuses GPU")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model, tokenizer = trainer.build_model(str(base), hp)
    if not trainer.TOKENIZER_VECTORS.exists():
        raise ValueError("primary conformance vectors missing")
    trainer.check_tokenizer(tokenizer)
    node = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    node.no_padding()
    node.no_truncation()
    for rows in splits.values():
        for row in rows:
            for side in ("chosen", "rejected"):
                text = render(row["prompt"], row[side])
                ids = node.encode(text).ids
                if len(ids) > 256 or tokenizer(text, truncation=False)["input_ids"] != ids:
                    raise ValueError("literal training/node tokenizer mismatch")
    frozen_raw = (HERE / "data/frozen-controls.json").read_bytes()
    if sha(frozen_raw) != contract["frozen_controls_sha256"]:
        raise ValueError("immutable training-only baseline changed")
    frozen = json.loads(frozen_raw)
    controls = replay(splits["validation"], frozen, node)
    if max(controls.values()) > .65:
        raise ValueError("validation shortcut gate failed")
    policy = module("research_evaluation_policy", SHARED / "qa-experiment/training_policy.py")
    def evaluate(rows):
        with policy.isolated_evaluation(model, torch, np):
            return trainer.evaluate(model, tokenizer, rows, hp)
    def trainable_hash():
        digest = __import__("hashlib").sha256()
        for name, value in model.named_parameters():
            if value.requires_grad:
                digest.update(name.encode())
                digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
    def save(path):
        path.mkdir(exist_ok=True)
        model.save_pretrained(path)
        trainer.save_reward_head(model, path)
        original = torch.load(path / "reward_head.pt", weights_only=True, map_location="cpu")
        for artifact in list(path.glob("*.safetensors")) + [path / "reward_head.pt"]:
            stamp(artifact)
        restored = torch.load(path / "reward_head.pt", weights_only=True, map_location="cpu")
        if original.keys() != restored.keys() or any(not torch.equal(original[k], restored[k]) for k in original):
            raise ValueError("licence stamp changed reward head")
        preserve(root)
    initial_hash = trainable_hash()
    save(root / "initial")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=hp.lr, weight_decay=hp.weight_decay)
    sampler = torch.Generator().manual_seed(42)
    loader = DataLoader(splits["train"], batch_size=1, shuffle=True, collate_fn=trainer.PairCollator(tokenizer, 256),
                        generator=sampler, num_workers=0)
    best, best_state, history, timings = None, None, [], []
    for epoch in range(1, 4):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_start = time.monotonic()
        for index, batch in enumerate(loader):
            c = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            r = trainer._reward(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
            loss = trainer.bt_loss(c, r, batch["margin"])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            (loss / 16).backward()
            if (index + 1) % 16 == 0:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                timings.append(time.monotonic() - step_start)
                save(root / "checkpoint")
                write_json(root / "checkpoint.json", {"optimizer_steps": len(timings), "epoch": epoch, "step_seconds": timings,
                           "pending": "validation-only selection then one fixed audit", "commit": os.environ["GITHUB_SHA"], "run_id": os.environ["GITHUB_RUN_ID"]})
                remaining = 24 - len(timings)
                if time.monotonic() - started + remaining * max(timings) > 220 * 60:
                    raise TimeoutError("measured CPU throughput exceeds training reserve")
                step_start = time.monotonic()
        development = evaluate(splits["validation"])
        best, improved = policy.development_selection(best, epoch, development)
        if improved:
            best_state = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
            best_hash = trainable_hash()
            save(root / "adapter")
        history.append({"epoch": epoch, "validation": development, "selected": improved})
        write_json(root / "development.json", {"best": best, "history": history, "audit_scored": False})
    if best_state is None or len(timings) != 24:
        raise ValueError("fixed three-epoch recipe incomplete")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(best_state[name])
    if trainable_hash() != best_hash:
        raise ValueError("development checkpoint restoration failed")
    save(root / "adapter")
    trained = evaluate(splits["audit"])
    record_owner_audit(model, tokenizer, trainer, splits["audit"])
    ablated = [{**r, "prompt": "Claim: " + resolve_claim(r)} for r in splits["audit"]]
    ablation = evaluate(ablated)["pairwise_accuracy"]
    native = []
    model.eval()
    with torch.no_grad():
        for row in splits["audit"]:
            for side in ("chosen", "rejected"):
                encoded = tokenizer(render(row["prompt"], row[side]), truncation=False, return_tensors="pt")
                native.append(trainer._reward(model, encoded["input_ids"], encoded["attention_mask"]).item())
    wins = [native[2 * i] > native[2 * i + 1] for i in range(64)]
    group_wins = [wins[i] and wins[i + 1] for i in range(0, 64, 2)]
    subsets = {"original_claim": sum(wins[::2]) / 32, "opposite_direction": sum(wins[1::2]) / 32}
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import metadata, target_dims, tensor_names, write_package
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(base, local_files_only=True)
    tensors, params = read_peft(root / "adapter", cfg)
    foundation_spec = module("research_export_foundation", SHARED / "prepare.py")
    hashes = foundation_spec.FOUNDATION
    write_json(root / "foundation.json", {"foundation_hash": hashes["foundation_lora_fp32.onnx"], "foundation_version": "modernbert-base-lora16-v1",
               "tokenizer_hash": TOKENIZER_SHA, "base_model": str(base), "r_max": 16, "num_layers": cfg.num_hidden_layers,
               "hidden_size": cfg.hidden_size, "targets": {key: list(value) for key, value in target_dims(cfg).items()}})
    meta = metadata(cls="judgment", intent=FAMILY, foundation_hash=hashes["foundation_lora_fp32.onnx"], foundation_version="modernbert-base-lora16-v1",
                    rank=params["rank"], alpha=params["alpha"], target_modules=params["target_modules"], max_tokens=256,
                    training={"family": FAMILY, "children": [{"intent": CHILD, "dataset_version": contract["splits"]["audit"]["sha256"],
                              "pairs": 64, "accuracy": trained["pairwise_accuracy"], "held_out": True}],
                              "trainer": "source-excerpt fixed CPU experiment", "backbone": "ModernBERT-base",
                              "backbone_version": "8949b909ec900327062f0ebf497f51aef5e6f0c8"})
    meta.update(licence_metadata())
    meta["x-original-groups"] = "32 independent audit groups, 64 correlated mirrored query pairs"
    meta["x-supported-formats"] = "prose,json"
    write_package(root / (FAMILY + ".safetensors"), meta, {name: tensors[name] for name in tensor_names(cfg.num_hidden_layers)})
    stamp(root / (FAMILY + ".safetensors"))
    del model, optimizer, parameters, loader, tensors, best_state, c, r, loss
    gc.collect()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model, tokenizer = trainer.build_model(str(base), hp)
    if trainable_hash() != initial_hash:
        raise ValueError("seed-matched untrained initialization differs")
    untrained = evaluate(splits["audit"])
    record_owner_untrained(model, tokenizer, trainer, splits["audit"], untrained)
    del model
    gc.collect()
    audit_controls = replay(splits["audit"], frozen, node)
    checks = quality(trained["pairwise_accuracy"], untrained["pairwise_accuracy"], audit_controls, ablation)
    if min(subsets.values()) < .70:
        checks["passed"] = False
        checks["failures"].append("claimed direction subset below 70 percent")
    symbolic = sum(symbolic_score(r["prompt"], r["chosen"]) > symbolic_score(r["prompt"], r["rejected"]) for r in splits["audit"]) / 64
    runtime = runtime_check(root, kit, splits["audit"], native)
    passed = checks["passed"] and runtime
    write_json(root / "result.json", {"status": "SOURCE_CANDIDATE_NODE_UNTESTED" if passed else "FAILED_CANDIDATE",
               "family": FAMILY, "child": CHILD, "contract": contract, "trained": trained, "untrained": untrained,
               "frozen_shortcuts": audit_controls, "evidence_removed_accuracy": ablation, "symbolic_task_baseline_accuracy": symbolic,
               "quality": checks, "runtime": runtime, "selected_epoch": best["epoch"], "selected_state_sha256": best_hash,
               "independent_audit_groups": 32, "pair_count": 64, "elapsed_seconds": time.monotonic() - started,
               "direction_subsets": subsets, "groups_with_both_variants_correct": sum(group_wins),
               "runtime_engine_sha256": sha((foundation / "foundation_engine.wasm").read_bytes()),
               "run_id": os.environ["GITHUB_RUN_ID"], "commit": os.environ["GITHUB_SHA"], "node_identity": "https://devnode.telegraphprotocol.com",
               "initial_state_sha256": initial_hash, "node_qualification": "not attempted", "canonical_claim": False})
    return 0 if passed else 1


def resolve_claim(row):
    return row["prompt"].split("\nClaim: ", 1)[1]


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    if not root.is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("remote output must be inside runner temporary storage")
    def stop(signum, frame):
        raise TimeoutError("bounded CPU job stopped: " + str(signum))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(300 * 60)
    try:
        return experiment(root)
    except Exception as exc:
        write_json(root / "failure.json", {"status": "INCOMPLETE", "error": type(exc).__name__ + ": " + str(exc)})
        raise
    finally:
        preserve(root)


if __name__ == "__main__":
    raise SystemExit(main())
