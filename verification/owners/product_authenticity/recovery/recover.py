# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only, byte-preserving recovery of the already selected broad delta."""
import argparse
from collections import Counter
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
DEV = HERE.parent / "development"
sys.path.insert(0, str(DEV))
sys.path.insert(0, str(HERE.parent / "broad_development"))
from runtime_contract import execute, remote_only, score_value
from remote_develop import class_snapshot, fetch, metrics, write
from broad_data import CHILD, FAMILY, FORMATS, build, digest, render, source_controls

PACKAGE_SHA = "6359621587da0a079113a7f2912ce059aab5cd267719bea077fae0ac0c00e84b"
ORIGINAL_RUN = 44630753101
ORIGINAL_COMMIT = "d7d21aa4c46d2fffc635a1e93fdbf04fc687f860"


def frozen_answer_control(np, rows, path, train_sha):
    import re
    frozen = json.loads(path.read_text())
    if frozen["fitted_on"] != "train_only" or frozen["train_sha256"] != train_sha:
        raise ValueError("original train-fitted answer baseline binding mismatch")
    vocabulary = frozen["vocabulary"]
    weights = np.array(frozen["weight_bits"], dtype="<u4").view("<f4")
    bias = np.array([frozen["bias_bits"]], dtype="<u4").view("<f4")[0]
    if len(weights) != len(vocabulary) or not np.isfinite(weights).all() or not np.isfinite(bias):
        raise ValueError("invalid original immutable baseline coefficients")
    def score(answer):
        words = re.findall(r"[a-z0-9]+", answer.lower())
        counts = Counter(words + [a + " " + b for a, b in zip(words, words[1:])])
        vector = np.zeros(len(vocabulary), dtype=np.float32)
        for word, count in counts.items():
            if word in vocabulary:
                vector[vocabulary[word]] = count
        norm = np.linalg.norm(vector)
        vector = vector / max(float(norm), 1.0)
        return float(vector @ weights + bias)
    return {style: metrics([score(r[side]) for r in rows if r["format"] == style for side in ("chosen", "rejected")],
                           [r for r in rows if r["format"] == style]) for style in FORMATS}


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    package = HERE / "PRODUCT_AUTHENTICITY.safetensors"
    if digest(package.read_bytes()) != PACKAGE_SHA:
        raise ValueError("selected package bytes changed before recovery")
    sources = json.loads((DEV / "model-sources.json").read_text())
    data, splits = build(root / "data")
    original_data = json.loads((HERE / "original-data-report.json").read_text())
    if data["files"] != original_data["files"] or data["normalized_overlap"] != 0:
        raise ValueError("recovery must reuse the exact original data, not new selection data")
    write(root / "recovery-report.json", {"schema_version": 1, "status": "RECOVERY_INCOMPLETE", "family": FAMILY,
          "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
          "original_run_id": ORIGINAL_RUN, "original_commit": ORIGINAL_COMMIT, "package_sha256": PACKAGE_SHA,
          "training_performed": False, "audit_pairs": 0, "registration_ready": False})
    archive = root / "kit.tar.gz"
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + sources["kit"]["revision"], archive, sources["kit"]["sha256"], 32 << 20)
    with tarfile.open(archive) as source:
        if sum(m.size for m in source.getmembers()) > 128 << 20:
            raise ValueError("primary source expansion ceiling")
        source.extractall(root, filter="data")
    kit = root / ("telegraph-wasm-baseline-" + sources["kit"]["revision"])
    subprocess.run(["go", "test", "-mod=readonly", str(DEV / "packagecheck.go"), str(DEV / "packagecheck_test.go")], cwd=kit, check=True, timeout=600)
    for name, source in (("packagecheck", str(DEV / "packagecheck.go")), ("adapterexec", "./cmd/adapterexec")):
        subprocess.run(["go", "build", "-mod=readonly", "-trimpath", "-o", str(root / name), source], cwd=kit, check=True, timeout=600)
    sys.path.insert(0, str(kit))
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from float_runtime import train_adapter as trainer
    from float_runtime.lora_format import read_package, target_dims, tensor_names
    from float_runtime.export_adapter import read_peft
    from licensing import preserve_artifact
    if torch.cuda.is_available():
        raise RuntimeError("GPU recovery is not allowed")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    foundation, base = root / "foundation", root / "base"
    foundation.mkdir(exist_ok=True)
    base.mkdir(exist_ok=True)
    for name, expected in sources["foundation"].items():
        fetch(sources["foundation_release"] + name, foundation / name, expected, 2 << 30)
    for name, expected in sources["base"]["files"].items():
        fetch("https://huggingface.co/" + sources["base"]["id"] + "/resolve/" + sources["base"]["revision"] + "/" + name, base / name, expected, 2 << 30)
    (base / "tokenizer.json").write_bytes((foundation / "tokenizer.json").read_bytes())
    os.environ["HF_HUB_OFFLINE"] = "1"
    meta, tensors = read_package(package)
    if meta["max_tokens"] != "512" or meta["intent"] != FAMILY or meta["foundation_hash"] != sources["foundation"]["foundation_lora_fp32.onnx"]:
        raise ValueError("selected package identity/layout changed")
    hp = trainer.Hyperparams(max_length=512, batch_size=1, grad_accum=8, epochs=3, eval_batch=2,
                            lora_rank=int(meta["rank"]), lora_alpha=float(meta["alpha"]), lora_dropout=0.0, grad_checkpointing=False,
                            gate_min_acc=.8, early_stopping=False, lr=4e-5)
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    model, tokenizer = trainer.build_model(str(base), hp)
    trainer.check_tokenizer(tokenizer)
    node = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    node.no_padding()
    node.no_truncation()
    for row in splits["validation"]:
        for side in ("chosen", "rejected"):
            value = render(row["prompt"], row[side])
            ids = node.encode(value).ids
            if len(ids) > 512 or tokenizer(value, truncation=False)["input_ids"] != ids:
                raise ValueError("recovery tokenizer mismatch/truncation")
    def evaluate(rows):
        model.eval()
        values = []
        with torch.no_grad():
            for row in rows:
                for side in ("chosen", "rejected"):
                    encoded = tokenizer(render(row["prompt"], row[side]), truncation=False, return_tensors="pt")
                    values.append(trainer._reward(model, encoded["input_ids"], encoded["attention_mask"]).item())
                if time.monotonic() - started > 300 * 60:
                    raise TimeoutError("bounded recovery reserve exceeded")
        return metrics(values, rows), values
    untrained = {style: evaluate([r for r in splits["validation"] if r["format"] == style])[0] for style in FORMATS}
    # Invert the pinned exporter, copying all canonical package tensors. Never
    # use the last-epoch checkpoint in place of the validation-selected package.
    parameters = dict(model.named_parameters())
    used = []
    with torch.no_grad():
        for name, value in tensors.items():
            if name.startswith("lora."):
                _, layer, section, target, matrix = name.split(".")
                key = f"base_model.model.model.layers.{layer}.{section}.{target}.lora_{matrix}.default.weight"
                incoming = torch.from_numpy(value.T.copy())
            else:
                key = "base_model.model.classifier.modules_to_save.default." + name.split(".")[1]
                incoming = torch.from_numpy(value.T.copy() if name == "head.weight" else value.copy())
            if key not in parameters or parameters[key].shape != incoming.shape:
                raise ValueError("exact exporter inverse shape/key mismatch: " + name)
            parameters[key].copy_(incoming)
            used.append(key)
    restored = root / "restored"
    restored.mkdir(exist_ok=True)
    model.save_pretrained(restored)
    trainer.save_reward_head(model, restored)
    for path in [*restored.glob("*.safetensors"), restored / "reward_head.pt"]:
        preserve_artifact(path)
    rebuilt, layout = read_peft(restored, model.config)
    expected_names = tensor_names(model.config.num_hidden_layers)
    if set(tensors) != set(expected_names) or len(used) != len(expected_names) or any(rebuilt[n].tobytes() != tensors[n].tobytes() for n in expected_names):
        raise ValueError("exporter inverse did not preserve every selected tensor byte")
    write(root / "tensor-roundtrip.json", {"package_sha256": PACKAGE_SHA, "tensors": len(expected_names), "all_tensor_bytes_identical": True,
          "rank": layout["rank"], "training_performed": False})
    validation, actual_scores = {}, []
    for style in FORMATS:
        rows = [r for r in splits["validation"] if r["format"] == style]
        validation[style], values = evaluate(rows)
        for view in ("direct", "typed"):
            selected = [r for r in rows if r["view"] == view]
            selected_scores = [s for i, row in enumerate(rows) if row["view"] == view for s in values[2*i:2*i+2]]
            validation[style].setdefault("per_view", {})[view] = metrics(selected_scores, selected)
        actual_scores.append({"format": style, "values": values})
    write(root / "native-validation-scores.json", {"package_sha256": PACKAGE_SHA, "scores": actual_scores})
    train_sha = digest((root / "data/train.jsonl").read_bytes())
    answer_baseline = frozen_answer_control(np, splits["validation"], HERE / "frozen-answer-only-baseline.json", train_sha)
    token_baseline = json.loads((HERE / "frozen-token-length-baseline.json").read_text())
    if token_baseline["fitted_on"] != "train_only" or token_baseline["train_sha256"] != train_sha or token_baseline["orientation"] not in (-1, 1):
        raise ValueError("original token baseline binding mismatch")
    controls = source_controls(splits["validation"])
    for style in FORMATS:
        rows = [r for r in splits["validation"] if r["format"] == style]
        counts = [token_baseline["orientation"] * len(node.encode(render(r["prompt"], r[s])).ids) for r in rows for s in ("chosen", "rejected")]
        controls[style].update(train_fitted_answer_only=answer_baseline[style]["accuracy"], train_fitted_token_length=metrics(counts, rows)["accuracy"], prompt_only=.5)
    # One actual pair for every claimed response format, view and comparison kind.
    fixed_rows, seen = [], set()
    for row in splits["validation"]:
        key = (row["format"], row["view"], row["kind"])
        if key not in seen:
            seen.add(key)
            fixed_rows.append(row)
    fixed = [node.encode(render(r["prompt"], r[side])).ids for r in fixed_rows for side in ("chosen", "rejected")]
    _, native_fixed = evaluate(fixed_rows)
    cfg = model.config
    model = None
    gc.collect()
    spec = {"foundation_hash": meta["foundation_hash"], "foundation_version": meta["foundation_version"], "tokenizer_hash": meta["tokenizer_hash"],
            "r_max": 16, "num_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size, "targets": {key: list(value) for key, value in target_dims(cfg).items()}}
    write(root / "foundation.json", spec)
    snapshot = class_snapshot(root)
    command = [str(root / "packagecheck"), "--intent", FAMILY, "--child", CHILD, "--expected-max-tokens", "512",
               "--class-snapshot", str(root / "class-snapshot.json"), "--snapshot-sha256", snapshot,
               "--package", str(package), "--sha256", PACKAGE_SHA, "--foundation", str(root / "foundation.json"),
               "--foundation-sha256", digest((root / "foundation.json").read_bytes()), "--out", str(root)]
    parsed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    write(root / "parser-result.json", {"exit": parsed.returncode, "stdout": parsed.stdout, "stderr": parsed.stderr})
    if parsed.returncode:
        raise ValueError("full primary parser or frozen-token binding still failed")
    manifest = json.loads(parsed.stdout)
    bare = manifest["keccak256"].removeprefix("0x")
    identity = {"foundation_hash": meta["foundation_hash"], "engine_hash": sources["foundation"]["foundation_engine.wasm"], "tokenizer_hash": meta["tokenizer_hash"], "adapter_hash": bare}
    runtime = execute(root, package, identity, fixed, max_tokens=512)
    reference = [score_value(v["score_bits"]) for v in runtime["vectors"]]
    if len(reference) != len(native_fixed) or any(abs(a-b) > .001 + .001*abs(a) for a, b in zip(native_fixed, reference)):
        raise ValueError("native/reference fixed-vector mismatch")
    if any((native_fixed[i] > native_fixed[i+1]) != (reference[i] > reference[i+1]) for i in range(0, len(reference), 2)):
        raise ValueError("native/reference ordering mismatch")
    if digest(package.read_bytes()) != PACKAGE_SHA:
        raise ValueError("recovery changed selected package bytes")
    quality = {style: validation[style]["accuracy"] >= .8 and min(validation[style]["per_label"].values()) >= .7
               and validation[style]["accuracy"] + 1e-9 >= untrained[style]["accuracy"] + .05
               and validation[style]["accuracy"] + 1e-9 >= max(controls[style].values()) + .10 for style in FORMATS}
    write(root / "recovery-report.json", {"schema_version": 1, "status": "DEVELOPMENT_RECOVERED", "family": FAMILY,
          "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"], "original_run_id": ORIGINAL_RUN,
          "original_commit": ORIGINAL_COMMIT, "package_sha256": PACKAGE_SHA, "package_keccak": bare, "package_bytes": package.stat().st_size,
          "data": data, "validation": validation, "untrained_validation": untrained, "shortcuts": controls, "runtime": runtime,
          "training_performed": False, "tensor_roundtrip_passed": True, "package_bits_unchanged": True,
          "development_quality_passed": all(quality.values()), "development_quality_by_format": quality,
          "audit_pairs": 0, "audit_used_for_selection": False, "registration_ready": False,
          "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy", "safetensors")},
          "limits": ["This is byte-preserving integration recovery and development validation, not independent audit or canonical qualification.",
                     "The original validation selected the package. No new candidate or checkpoint is selected here.",
                     "Baseline coefficients and token direction come from the original train-fitted artifacts, without fitting again."]})


if __name__ == "__main__":
    main()
