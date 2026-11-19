# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded CPU training and package/runtime freeze. Audit remains separate."""
import argparse
import datetime as dt
import gc
import importlib.metadata
import json
import os
import random
import socket
import shutil
import struct
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

from contracts import (BASE_FILES, BASE_REVISION, FAMILY, FORMATS, FOUNDATION, KIT, KIT_SHA,
                       LABELS, basic_margins, margin_accuracy, read_development, remote_only,
                       render, sha, validation_gate, words, write_json)
from training_record import development_record
from class_binding import collect_classes

HERE = Path(__file__).resolve().parent
RELEASE = "https://github.com/telegraphprotocol/telegraph-foundation/releases/download/modernbert-base-lora16-v1/"


def fetch(url, target, digest, ceiling=1 << 30):
    remote_only()
    request = urllib.request.Request(url, headers={"User-Agent": "Threat-IP-owned-CPU-learning/1.0"})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=45) as source, target.open("wb") as out:
        total = 0
        while chunk := source.read(1 << 20):
            total += len(chunk)
            if total > ceiling or time.monotonic() - start > 600:
                raise ValueError("public model/source download ceiling")
            out.write(chunk)
    if sha(target.read_bytes()) != digest:
        raise ValueError("public model/source hash mismatch: " + target.name)


def prepare(root):
    remote_only()
    foundation, base = root / "foundation", root / "base"
    foundation.mkdir(exist_ok=True)
    base.mkdir(exist_ok=True)
    for name, digest in FOUNDATION.items():
        fetch(RELEASE + name, foundation / name, digest)
    for name, digest in BASE_FILES.items():
        fetch("https://huggingface.co/answerdotai/ModernBERT-base/resolve/" + BASE_REVISION + "/" + name, base / name, digest)
    (base / "tokenizer.json").write_bytes((foundation / "tokenizer.json").read_bytes())
    archive = root / "kit.tar.gz"
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + KIT, archive, KIT_SHA, 32 << 20)
    with tarfile.open(archive) as source:
        if sum(member.size for member in source.getmembers()) > 128 << 20:
            raise ValueError("reference source inventory exceeds ceiling")
        source.extractall(root, filter="data")
    kit = root / ("telegraph-wasm-baseline-" + KIT)
    sys.path.insert(0, str(kit))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return foundation, base, kit


def fit_controls(rows, root):
    remote_only()
    # This function runs remotely only. Save fitted bytes once; audit must read
    # these coefficients, never refit them on another machine or split.
    import numpy as np
    vocabulary = sorted({word for row in rows for side in ("chosen", "rejected") for word in words(row[side])})
    lookup = {word: i for i, word in enumerate(vocabulary)}
    x = np.zeros((len(rows), len(vocabulary)), dtype=np.float64)
    for i, row in enumerate(rows):
        for side, sign in (("chosen", 1), ("rejected", -1)):
            for word in words(row[side]):
                x[i, lookup[word]] += sign
    weights = np.zeros(len(vocabulary), dtype=np.float64)
    for _ in range(80):
        z = np.clip(x @ weights, -30, 30)
        p = 1 / (1 + np.exp(-z))
        weights -= 0.2 * ((x.T @ (p - 1)) / len(rows) + 0.001 * weights)
    controls = {"train_sha256": sha((HERE / "data/train.jsonl").read_bytes()), "vocabulary": vocabulary,
                "answer_only_coefficients": weights.tolist(), "basic_coefficients": {"length": 1, "overlap": 1, "prefer_current": 1},
                "fit_split": "train", "fit_method": "80 fixed full-batch logistic steps, float64, no audit data"}
    write_json(root / "frozen-controls.json", controls)
    return controls


def shortcuts(rows, controls):
    lookup = dict(zip(controls["vocabulary"], controls["answer_only_coefficients"]))
    margins = {key: [] for key in ("length", "overlap", "prefer_current", "answer_only")}
    for row in rows:
        for name, value in basic_margins(row).items():
            margins[name].append(value)
        margins["answer_only"].append(sum(lookup.get(word, 0) for word in words(row["chosen"])) - sum(lookup.get(word, 0) for word in words(row["rejected"])))
    return {name: margin_accuracy(values) for name, values in margins.items()}


def engine_vectors(root, foundation, package, manifest, tokenizer, rows, full_validation=False):
    remote_only()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {**os.environ, "FOUNDATION_MODEL_PATH": str(foundation / "foundation_lora_fp32.onnx"),
           "FOUNDATION_MODEL_HASH": FOUNDATION["foundation_lora_fp32.onnx"],
           "FOUNDATION_ENGINE_WASM_PATH": str(foundation / "foundation_engine.wasm"),
           "FOUNDATION_ENGINE_WASM_HASH": FOUNDATION["foundation_engine.wasm"], "FOUNDATION_POOL_SIZE": "1"}
    adapter_hash = manifest["keccak256"].removeprefix("0x")
    identity = {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"], "tokenizer_hash": FOUNDATION["tokenizer.json"], "adapter_hash": adapter_hash}
    url = "http://127.0.0.1:" + str(port) + "/runnable"
    def post(body):
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            raw = response.read((1 << 20) + 1)
            if len(raw) > 1 << 20:
                raise ValueError("reference response ceiling")
            return json.loads(raw)
    with (root / "reference-engine.log").open("w") as log:
        process = subprocess.Popen([str(root / "adapterexec"), "--package", str(package), "--listen", "127.0.0.1:" + str(port)],
                                   env=env, stdout=log, stderr=log)
        try:
            for _ in range(90):
                if process.poll() is not None:
                    raise RuntimeError("actual reference engine exited")
                try:
                    ready = post({**identity, "batch": []})
                    if any(ready.get(key) != value for key, value in identity.items()):
                        raise ValueError("literal reference identity echo mismatch")
                    break
                except OSError:
                    time.sleep(2)
            else:
                raise RuntimeError("reference engine readiness deadline")
            controls = []
            for field in identity:
                try:
                    post({**identity, field: "0" * 64, "batch": []})
                except urllib.error.HTTPError as exc:
                    if exc.code != 400:
                        raise ValueError("identity refusal differs from literal HTTP contract") from exc
                    controls.append(field)
                else:
                    raise ValueError("reference accepted a wrong pinned identity")
            selected = rows if full_validation else rows[:4]
            vectors = [tokenizer.encode(render(row["prompt"], row[side])).ids for row in selected for side in ("chosen", "rejected")]
            bits = []
            for start in range(0, len(vectors), 8):
                batch = vectors[start:start + 8]
                first = post({**identity, "batch": [{"tokens": tokens} for tokens in batch]})
                if any(first.get(key) != value for key, value in identity.items()) or len(first.get("results", [])) != len(batch):
                    raise ValueError("reference literal identity or result count differs")
                bits.extend(value["score"] for value in first["results"])
            repeated = post({**identity, "batch": [{"tokens": tokens} for tokens in vectors[:8]]})
            if [value["score"] for value in repeated.get("results", [])] != bits[:8] or any(repeated.get(key) != value for key, value in identity.items()):
                raise ValueError("reference repeated bits or identity differs")
            if any(type(value) is not int or not 0 <= value < 1 << 32 or value & 0x7f800000 == 0x7f800000 for value in bits):
                raise ValueError("reference returned invalid finite FP32 bits")
            proof = {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
                     "negative_identity_controls": len(controls), "vectors": [{"tokens": tokens, "score_bits": value} for tokens, value in zip(vectors[:8], bits[:8])],
                     "scope": "validation fixed vectors, not an independent quality audit"}
            if full_validation:
                scores = [struct.unpack("<f", struct.pack("<I", value))[0] for value in bits]
                wins = [scores[2*i] > scores[2*i+1] for i in range(len(selected))]
                metrics = {}
                for fmt in FORMATS:
                    indices = [i for i, row in enumerate(selected) if row["format"] == fmt]
                    metrics[fmt] = {"accuracy": sum(wins[i] for i in indices) / len(indices),
                                    "per_label": {label: sum(wins[i] for i in indices if selected[i]["label"] == label) /
                                                  sum(selected[i]["label"] == label for i in indices) for label in LABELS}}
                proof["validation_formats"] = metrics
                proof["validation_pairs"] = len(selected)
            return proof
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def main():
    remote_only()  # Must precede every neural import, fetch or model operation.
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    splits = read_development(HERE / "data")
    report = {"schema_version": 1, "family": FAMILY, "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
              "status": "TRAINING_IN_PROGRESS", "source_manifest_sha256": sha((HERE / "data/development-report.json").read_bytes()),
              "audit_used_for_selection": False, "audit_pairs": 0, "qualification_ready": False}
    write_json(root / "candidate-report.json", report)
    foundation, base, kit = prepare(root)
    import numpy as np
    import torch
    from float_runtime import train_adapter as trainer
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import metadata, target_dims, tensor_names, write_package
    from tokenizers import Tokenizer
    from transformers import AutoConfig, AutoTokenizer
    if torch.cuda.is_available():
        raise RuntimeError("this pipeline is CPU-only")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    hp = trainer.Hyperparams(max_length=512, batch_size=1, grad_accum=16, epochs=3, eval_batch=2, gate_min_acc=0.8,
                            gate_min_margin_over_baseline=0.1, grad_checkpointing=True)
    gate = trainer.gate_dataset(splits["train"], hp)
    write_json(root / "training-data-gate.json", gate)
    if not gate["ok"]:
        report.update(status="DATA_GATE_FAILED", data_gate=gate)
        write_json(root / "candidate-report.json", report)
        raise RuntimeError("unchanged upstream training-data gate refused the frozen corpus")
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    node_tok = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    node_tok.no_padding()
    node_tok.enable_truncation(max_length=512)
    checked = 0
    for rows in splits.values():
        for row in rows:
            pair_ids = []
            for side in ("chosen", "rejected"):
                text = render(row["prompt"], row[side])
                ids = tok(text, truncation=False)["input_ids"]
                if len(ids) > 512 or ids != node_tok.encode(text).ids or text != trainer.render(row["prompt"], row[side]):
                    raise ValueError("actual tokenizer/render/token ceiling conformance failed")
                pair_ids.append(ids)
                checked += 1
            if pair_ids[0] == pair_ids[1]:
                raise ValueError("different source claims became identical token vectors")
    trainer.check_tokenizer(tok)
    write_json(root / "tokenizer-check.json", {"rendered_texts": checked, "tokenizer_sha256": FOUNDATION["tokenizer.json"], "max_tokens": 512, "passed": True})
    controls = fit_controls(splits["train"], root)
    shortcut_values = {fmt: shortcuts([row for row in splits["validation"] if row["format"] == fmt], controls) for fmt in FORMATS}
    strongest = {fmt: max(values.values()) for fmt, values in shortcut_values.items()}
    def evaluate(model):
        values = {}
        for fmt in FORMATS:
            rows = [row for row in splits["validation"] if row["format"] == fmt]
            overall = trainer.evaluate(model, tok, rows, hp)
            values[fmt] = {"accuracy": overall["pairwise_accuracy"], "per_label": {label: trainer.evaluate(model, tok, [row for row in rows if row["label"] == label], hp)["pairwise_accuracy"] for label in LABELS}}
        return values
    licences = json.loads((HERE / "licences.json").read_text())
    cfg = AutoConfig.from_pretrained(base, local_files_only=True)
    def export(directory, output, trained, measured):
        tensors, params = read_peft(directory, cfg)
        meta = metadata(cls="judgment", intent=FAMILY, foundation_hash=FOUNDATION["foundation_lora_fp32.onnx"], foundation_version="modernbert-base-lora16-v1",
                        rank=params["rank"], alpha=params["alpha"], target_modules=params["target_modules"], max_tokens=512,
                        training=development_record(measured, len(splits["validation"])))
        meta["x-development-provenance"] = json.dumps({"trained": trained, "audit_required": True,
                                                       "train_sha256": sha((HERE / "data/train.jsonl").read_bytes()),
                                                       "validation_sha256": sha((HERE / "data/validation.jsonl").read_bytes()),
                                                       "scope": "source-specific observation expiry fidelity, not universal IP truth; development is not independent audit"})
        for key in ("notice", "apache_2_0", "kit_mit", "sand", "cc0", "source_notice"):
            meta["x-" + key.replace("_", "-")] = licences[key]
        write_package(output, meta, {name: tensors[name] for name in tensor_names(cfg.num_hidden_layers)})
        return meta
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    initial, _ = trainer.build_model(str(base), hp)
    untrained = evaluate(initial)
    seed_dir = root / "untrained-adapter"
    seed_dir.mkdir(exist_ok=True)
    initial.save_pretrained(seed_dir)
    trainer.save_reward_head(initial, seed_dir)
    export(seed_dir, root / "untrained.safetensors", False, untrained)
    del initial
    gc.collect()
    adapter = root / "trained-adapter"
    model, _, _ = trainer.train(str(base), splits["train"], splits["validation"], adapter, hp)
    trained = evaluate(model)
    (adapter / "tokenizer.json").write_bytes((foundation / "tokenizer.json").read_bytes())
    package = root / (FAMILY + ".safetensors")
    meta = export(adapter, package, True, trained)
    del model
    gc.collect()
    frozen_at = dt.datetime.now(dt.timezone.utc).isoformat()
    decision = validation_gate(trained, untrained, strongest)
    report.update(status="CANDIDATE_FROZEN_AUDIT_REQUIRED" if decision["passed"] else "VALIDATION_FAILED", frozen_at=frozen_at,
                  package_sha256=sha(package.read_bytes()), package_bytes=package.stat().st_size,
                  validation={"trained": trained, "untrained": untrained, "shortcuts": shortcut_values, "gate": decision},
                  frozen_controls_sha256=sha((root / "frozen-controls.json").read_bytes()), hyperparameters=asdict(hp),
                  versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy", "safetensors")},
                  runtime_node=socket.gethostname(), serving_commit=os.environ["GITHUB_SHA"])
    write_json(root / "candidate-report.json", report)  # Preserve actual metrics even if parser/runtime fails.
    for name in ("adapterexec", "adapterbuild"):
        subprocess.run(["go", "build", "-trimpath", "-o", str(root / name), "./cmd/" + name], cwd=kit, check=True, timeout=600)
    live_dir = kit / "cmd/threat-owner-livecheck"
    live_dir.mkdir(exist_ok=True)
    shutil.copyfile(HERE / "livecheck.go", live_dir / "main.go")
    subprocess.run(["go", "build", "-trimpath", "-o", str(root / "livecheck"), "./cmd/threat-owner-livecheck"], cwd=kit, check=True, timeout=600)
    collect_classes(root / "live-class-proof.json")
    spec = {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"], "foundation_version": "modernbert-base-lora16-v1", "tokenizer_hash": FOUNDATION["tokenizer.json"],
            "base_model": str(base), "r_max": 16, "num_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size, "targets": {key: list(value) for key, value in target_dims(cfg).items()}}
    write_json(root / "foundation.json", spec)
    parse = subprocess.run([str(root / "livecheck"), "--package", str(package), "--classes", str(root / "live-class-proof.json"),
                            "--foundation", str(root / "foundation.json"), "--out", str(root)], capture_output=True, text=True, timeout=60)
    write_json(root / "parser-result.json", {"exit": parse.returncode, "stdout": parse.stdout, "stderr": parse.stderr})
    if parse.returncode:
        raise RuntimeError("actual full parser refused package; exact stderr retained in parser-result.json")
    manifest = json.loads(parse.stdout)
    if manifest["sha256"] != sha(package.read_bytes()):
        raise ValueError("full package parser hash mismatch")
    write_json(root / "package-manifest.json", manifest)
    runtime = engine_vectors(root, foundation, package, manifest, node_tok, splits["validation"])
    write_json(root / "runtime-proof.json", runtime)
    decision = validation_gate(trained, untrained, strongest)
    report.update(status="CANDIDATE_FROZEN_AUDIT_REQUIRED" if decision["passed"] else "VALIDATION_FAILED", frozen_at=frozen_at,
                  package_sha256=sha(package.read_bytes()), package_keccak=manifest["keccak256"].removeprefix("0x"),
                  package_bytes=package.stat().st_size, foundation_sha256=FOUNDATION["foundation_lora_fp32.onnx"], engine_sha256=FOUNDATION["foundation_engine.wasm"],
                  tokenizer_sha256=FOUNDATION["tokenizer.json"], validation={"trained": trained, "untrained": untrained, "shortcuts": shortcut_values, "gate": decision},
                  frozen_controls_sha256=sha((root / "frozen-controls.json").read_bytes()), runtime=runtime, hyperparameters=asdict(hp),
                  versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy", "safetensors")},
                  runtime_node=socket.gethostname(), serving_commit=os.environ["GITHUB_SHA"])
    write_json(root / "candidate-report.json", report)
    print(json.dumps({"status": report["status"], "validation": report["validation"], "package_sha256": report["package_sha256"], "audit_pairs": 0, "qualification_ready": False}))
    if not decision["passed"]:
        raise RuntimeError("validation failed unchanged gates; audit remains untouched")


if __name__ == "__main__":
    main()
