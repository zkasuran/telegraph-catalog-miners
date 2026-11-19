# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Fixed remote CPU experiment. No registration or local model execution."""
import argparse
import ast
import gc
import math
import os
import random
import signal
import sys
import tarfile
import time
from dataclasses import asdict
from pathlib import Path

from data import HERE, digest, prepare, read_splits, write_json
from gates import run_gate
from licensing import OWN_ID, licence_metadata, preserve_artifact
from preserve import preserve
from runtime_check import (
    compile_tools,
    execute_runtime,
    model_remote_only,
    preflight,
    validate_package,
)
from static_check import main as check_static
from training_policy import EPOCHS, development_selection, isolated_evaluation

PREPARATION_SOURCE_SHA256 = "5b2ba64a11ff141ac96a20c491fc282f917e722183b2ab67c17e6a12c8ff519b"


def checked_preparation_ast(source):
    """Accept only the exact reviewed shared source before parsing any code."""
    if digest(source) != PREPARATION_SOURCE_SHA256:
        raise ValueError("reviewed shared preparation source hash mismatch")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare")
    return ast.Module(body=[function], type_ignores=[])


def remote_only():
    model_remote_only()
    if (os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_REPOSITORY") != "zkasuran/telegraph-catalog-miners"
            or not os.environ.get("RUNNER_TEMP")):
        raise RuntimeError("models run only on the assigned public GitHub CPU workflow")


def prepare_reference(root):
    """Execute the existing foundation/base preparation function unchanged."""
    import importlib.util
    shared = HERE.parent / "training"
    preparation = checked_preparation_ast((shared / "run.py").read_bytes())
    spec = importlib.util.spec_from_file_location("qa_shared_data", shared / "data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.insert(0, str(HERE.parent))
    from prepare import FOUNDATION, RELEASE
    env = {"FOUNDATION": FOUNDATION, "RELEASE": RELEASE, "SOURCES": module.SOURCES,
           "fetch": module.fetch, "tarfile": tarfile, "sys": sys, "os": os}
    # Only the function extracted from exact SHA-256-checked shared source executes.
    exec(compile(preparation, str(shared / "run.py"), "exec"), env)  # noqa: S102
    return env["prepare"](root)


def prepare_kit(root):
    """Source kit only, so a failed data gate does not fetch any weights."""
    import importlib.util
    shared = HERE.parent / "training"
    spec = importlib.util.spec_from_file_location("qa_source_fetch", shared / "data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pin = module.SOURCES["kit"]
    archive = root / "kit.tar.gz"
    module.fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + pin["revision"],
                 archive, pin["sha256"], 32 << 20)
    with tarfile.open(archive) as source:
        if sum(m.size for m in source.getmembers()) > 128 << 20:
            raise ValueError("source kit expansion ceiling")
        source.extractall(root, filter="data")
    return root / ("telegraph-wasm-baseline-" + pin["revision"])


def interval(wins, n):
    z = 1.959963984540054
    p = wins / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [center - half, center + half]


def quality(trained, untrained, controls):
    failures = []
    accuracy = trained["pairwise_accuracy"]
    values = [accuracy, untrained["pairwise_accuracy"], *controls.values(), *trained["subsets"].values()]
    if not controls or not trained["subsets"] or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        return {"passed": False, "failures": ["invalid or missing quality metric"]}
    if accuracy < 0.80:
        failures.append("audit ordering below 80 percent")
    if accuracy < untrained["pairwise_accuracy"] + 0.05:
        failures.append("less than five points above seed-matched untrained")
    if accuracy < max(controls.values()) + 0.10:
        failures.append("less than ten points above strongest frozen shortcut")
    if any(value > 0.65 for value in controls.values()):
        failures.append("audit shortcut exceeds 65 percent")
    if any(value < 0.70 for value in trained["subsets"].values()):
        failures.append("claimed length subset below 70 percent")
    return {"passed": not failures, "failures": failures}


def experiment(root):
    remote_only()
    started = time.monotonic()
    preserve(root)
    check_static()
    data_root = root / "data"
    prepare(data_root)
    kit = prepare_kit(root)
    trainer, hp, splits, controls, gate = run_gate(data_root, kit)
    data_report, _ = read_splits(data_root)
    write_json(root / "result.json", {"status": "INCOMPLETE", "phase": "data_gate", "dataset_gate": gate,
               "data": data_report, "runtime_integration": "INCOMPLETE"})
    if not data_report["training_allowed"] or not gate["ok"]:
        return 1
    # Bind the live primary family/child class before any model loading.
    compile_tools(root, kit)
    preflight(root)
    foundation, base, kit = prepare_reference(root)
    import numpy as np
    import torch
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import (
        metadata,
        target_dims,
        tensor_names,
        write_package,
    )
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoTokenizer
    if torch.cuda.is_available():
        raise RuntimeError("fixed CPU experiment refuses GPU runtime")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if not trainer.TOKENIZER_VECTORS.is_file():
        raise ValueError("missing primary tokenizer vectors")
    trainer.check_tokenizer(tok)
    from tokenizers import Tokenizer
    node = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    node.no_padding()
    node.no_truncation()
    for rows in splits.values():
        for row in rows:
            for side in ("chosen", "rejected"):
                text = trainer.render(row["prompt"], row[side])
                if tok(text, truncation=False)["input_ids"] != node.encode(text).ids:
                    raise ValueError("training and released tokenizer mismatch")

    def seed():
        random.seed(hp.seed)
        np.random.seed(hp.seed)
        torch.manual_seed(hp.seed)

    def state_hash(model):
        h = __import__("hashlib").sha256()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                h.update(name.encode())
                h.update(parameter.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def evaluate(model, rows):
        with isolated_evaluation(model, torch, np):
            metrics = trainer.evaluate(model, tok, rows, hp)
            metrics["subsets"] = {}
            for name, predicate in (("chosen_longer", lambda r: len(r["chosen"]) > len(r["rejected"])),
                                    ("chosen_shorter", lambda r: len(r["chosen"]) < len(r["rejected"]))):
                subset = [r for r in rows if predicate(r)]
                metrics["subsets"][name] = trainer.evaluate(model, tok, subset, hp)["pairwise_accuracy"]
            metrics["group_wilson_95"] = interval(round(metrics["pairwise_accuracy"] * len(rows)), len(rows))
        return metrics

    def save_checkpoint(model, destination):
        destination.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(destination)
        trainer.save_reward_head(model, destination)
        head = destination / "reward_head.pt"
        original = torch.load(head, map_location="cpu", weights_only=True)
        for path in sorted(destination.glob("*.safetensors")) + [head]:
            preserve_artifact(path)
        restored = torch.load(head, map_location="cpu", weights_only=True)
        if original.keys() != restored.keys() or any(not torch.equal(original[k], restored[k]) for k in original):
            raise ValueError("checkpoint ZIP licence record changed the reward head")
        preserve(root)

    def native_scores(model, rows):
        scores = []
        model.eval()
        with torch.no_grad():
            for row in rows:
                for side in ("chosen", "rejected"):
                    encoded = tok(trainer.render(row["prompt"], row[side]), truncation=False, return_tensors="pt")
                    value = trainer._reward(model, encoded["input_ids"], encoded["attention_mask"]).item()
                    if not math.isfinite(value):
                        raise ValueError("nonfinite native audit score")
                    scores.append(value)
        return scores

    seed()
    model, tok = trainer.build_model(str(base), hp)
    initial_hash = state_hash(model)
    save_checkpoint(model, root / "initial")
    # Complete all three fixed epochs. Only development loss selects the
    # checkpoint. Its extra evaluation consumes no training RNG.
    if hp.epochs != EPOCHS:
        raise ValueError("fixed three-epoch training contract changed")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=hp.lr, weight_decay=hp.weight_decay)
    sampler = torch.Generator(device="cpu").manual_seed(hp.seed)
    loader = DataLoader(splits["train"], batch_size=1, shuffle=True,
                        collate_fn=trainer.PairCollator(tok, 256), num_workers=0, generator=sampler)
    if len(loader) % hp.grad_accum:
        raise ValueError("fixed CPU batch requires complete accumulation groups")
    timings, losses = [], []
    best, best_state, best_hash = None, None, None
    development_history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_started = time.monotonic()
        for i, batch in enumerate(loader):
            rc = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rr = trainer._reward(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
            loss = trainer.bt_loss(rc, rr, batch["margin"])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            (loss / hp.grad_accum).backward()
            losses.append(loss.item())
            if (i + 1) % hp.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                timings.append(time.monotonic() - step_started)
                save_checkpoint(model, root / "checkpoint")
                write_json(root / "checkpoint.json", {"status": "INCOMPLETE", "optimizer_steps": len(timings),
                           "step_seconds": timings, "hyperparameters": asdict(hp), "initial_sha256": initial_hash,
                           "runtime_integration": "INCOMPLETE", "epoch": epoch, "max_epochs": EPOCHS,
                           "selected_development": best, "licence_metadata": licence_metadata()})
                remaining = EPOCHS * len(loader) / hp.grad_accum - len(timings)
                if time.monotonic() - started + remaining * max(timings) > 300 * 60:
                    raise TimeoutError("measured throughput exceeds fixed training reserve")
                step_started = time.monotonic()
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state().clone()
        sampler_rng = sampler.get_state().clone()
        development = evaluate(model, splits["validation"])
        numpy_after = np.random.get_state()
        if (random.getstate() != python_rng or numpy_rng[0] != numpy_after[0]
                or not np.array_equal(numpy_rng[1], numpy_after[1]) or numpy_rng[2:] != numpy_after[2:]
                or not torch.equal(torch_rng, torch.get_rng_state()) or not torch.equal(sampler_rng, sampler.get_state())):
            raise ValueError("development evaluation advanced training RNG")
        best, improved = development_selection(best, epoch, development)
        if improved:
            best_state = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
            best_hash = state_hash(model)
            save_checkpoint(model, root / "adapter")
        development_history.append({"epoch": epoch, "metrics": development, "selected": improved,
                                    "rng_restored": True, "split": "validation"})
        write_json(root / "development-selection.json", {"history": development_history, "best": best,
                   "best_trainable_sha256": best_hash, "selection_split": "validation", "selection_metric": "val_loss",
                   "audit_scored": False, "epochs_required": EPOCHS, "epochs_completed": epoch})
    if best_state is None or len(development_history) != EPOCHS:
        raise ValueError("three-epoch development selection did not complete")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(best_state[name])
    if state_hash(model) != best_hash:
        raise ValueError("restored checkpoint differs from the development-selected state")
    validation = development_history[best["epoch"] - 1]["metrics"]
    save_checkpoint(model, root / "adapter")
    tok.save_pretrained(root / "adapter")
    (root / "adapter/tokenizer.json").write_bytes((foundation / "tokenizer.json").read_bytes())
    trained = evaluate(model, splits["audit"])
    native = native_scores(model, splits["audit"])
    write_json(root / "native-audit-scores.json", {"scores": native, "audit_sha256": data_report["files"]["audit"]["sha256"],
               "scope": "same complete original rendering, native fp32 reference", "checkpoint_fixed_before_audit": True})
    cfg = AutoConfig.from_pretrained(base, local_files_only=True)
    tensors, params = read_peft(root / "adapter", cfg)
    sys.path.insert(0, str(HERE.parent))
    from prepare import FOUNDATION
    write_json(root / "foundation.json", {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
               "foundation_version": "modernbert-base-lora16-v1", "tokenizer_hash": FOUNDATION["tokenizer.json"],
               "base_model": str(base), "r_max": 16, "num_layers": cfg.num_hidden_layers,
               "hidden_size": cfg.hidden_size, "targets": {k: list(v) for k, v in target_dims(cfg).items()}})
    meta = metadata(cls="judgment", intent="QUESTION_ANSWERING", foundation_hash=FOUNDATION["foundation_lora_fp32.onnx"],
                    foundation_version="modernbert-base-lora16-v1", rank=params["rank"], alpha=params["alpha"],
                    target_modules=params["target_modules"], max_tokens=256,
                    training={"family": "QUESTION_ANSWERING", "children": [{"intent": "QUESTION_ANSWERING_GENERAL",
                              "dataset_version": data_report["files"]["audit"]["sha256"], "pairs": 64,
                              "accuracy": trained["pairwise_accuracy"], "held_out": True}],
                              "trainer": "OpenCode QA fixed source experiment", "backbone": "ModernBERT-base",
                              "backbone_version": "8949b909ec900327062f0ebf497f51aef5e6f0c8"})
    meta["x-source-provenance"] = __import__("json").dumps({"data": data_report, "hyperparameters": asdict(hp)}, separators=(",", ":"))
    meta.update(licence_metadata())
    write_package(root / "QUESTION_ANSWERING.safetensors", meta, {n: tensors[n] for n in tensor_names(cfg.num_hidden_layers)})
    preserve_artifact(root / "QUESTION_ANSWERING.safetensors")
    del model, tensors, optimizer, params, loader, sampler, rc, rr, loss, best_state
    gc.collect()
    seed()
    initial, _ = trainer.build_model(str(base), hp)
    if state_hash(initial) != initial_hash:
        raise ValueError("seed-matched untrained initialization mismatch")
    untrained = evaluate(initial, splits["audit"])
    del initial
    gc.collect()
    frozen_audit = controls(splits["audit"])
    report = {"status": "INCOMPLETE", "phase": "experiment_completed_runtime_pending", "family": "QUESTION_ANSWERING",
              "dataset_gate": gate, "data": data_report, "hyperparameters": asdict(hp), "trained": trained,
              "untrained": untrained, "validation": validation, "audit_controls": frozen_audit,
              "quality": quality(trained, untrained, frozen_audit), "initial_sha256": initial_hash,
              "selected_epoch": best["epoch"], "selection_split": "validation", "development_history": development_history,
              "selected_trainable_sha256": best_hash, "epochs_completed": EPOCHS, "evaluation_rng_restored": True,
              "licence_metadata": licence_metadata(), "own_additions_licence": OWN_ID,
              "optimizer_steps": len(timings), "step_seconds": timings, "train_loss_mean": sum(losses) / len(losses),
              "elapsed_seconds": time.monotonic() - started, "runtime_integration": "INCOMPLETE",
              "runtime_gap": "Full live-class parser and reference-runtime checks have not completed yet.",
              "node_qualification": "not attempted", "commit": os.environ["GITHUB_SHA"], "run_id": os.environ["GITHUB_RUN_ID"]}
    write_json(root / "result.json", report)
    manifest = validate_package(root)
    runtime = execute_runtime(root, manifest, native)
    passed = report["quality"]["passed"] and runtime["verdict"] == "PASS"
    report.update({"status": "EXPERIMENT_PASS_NODE_UNTESTED" if passed else "EXPERIMENT_FAIL",
                   "phase": "complete_source_experiment", "verdict": "PASS" if passed else "FAIL",
                   "runtime_integration": "PASS" if runtime["verdict"] == "PASS" else "FAIL",
                   "runtime": runtime, "package_manifest": manifest,
                   "elapsed_seconds": time.monotonic() - started})
    report.pop("runtime_gap", None)
    write_json(root / "result.json", report)
    if passed:
        write_json(root / "candidate-manifest.json", {"status": "SOURCE_CANDIDATE_PENDING_CONTROLLER_NODE_TEST",
                   "SPDX-License-Identifier": OWN_ID, "licence_metadata": licence_metadata(),
                   "family": "QUESTION_ANSWERING", "child": "QUESTION_ANSWERING_GENERAL", "package_sha256": manifest["sha256"],
                   "keccak256": manifest["keccak256"], "audit_sha256": data_report["files"]["audit"]["sha256"],
                   "commit": os.environ["GITHUB_SHA"], "run_id": os.environ["GITHUB_RUN_ID"],
                   "source_scope": data_report["scope"], "node_qualification": "not attempted", "registration": "not attempted"})
    preserve(root)
    return 0 if passed else 1


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if not args.root.resolve().is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("output must be inside runner temporary storage")
    def stop(signum, frame):
        raise TimeoutError("bounded job stopped by signal " + str(signum))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(300 * 60)
    try:
        return experiment(args.root)
    except Exception as exc:
        preserve(args.root)
        write_json(args.root / "failure.json", {"status": "INCOMPLETE", "error": type(exc).__name__ + ": " + str(exc),
                   "SPDX-License-Identifier": OWN_ID, "licence_metadata": licence_metadata()})
        raise
    finally:
        preserve(args.root)


if __name__ == "__main__":
    sys.exit(main())
