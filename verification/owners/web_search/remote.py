# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded public CPU training and reference-engine audit. Never local inference."""
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import random
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from build import render
from common import FOUNDATION, FOUNDATION_RELEASE, KIT, KIT_SHA, NODE, strict_json
from contracts import (FAMILY, HERE, REPO, bounded, literal_reply, metrics,
                       quality_gate, read_splits, ready_report, sha, shortcuts, write)
from licensing import licence_metadata, preserve_artifact
from proxy import Backend, BoundedServer, identity, input_root


def remote_only():
    if (os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_REPOSITORY") != REPO or
            os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch" or not os.environ.get("RUNNER_TEMP") or
            os.environ.get("OWNER_SOURCE_ONLY") == "1"):
        raise RuntimeError("models run only in the bounded public CPU workflow")


def check_reply(value, count, package, foundation):
    if foundation["foundation_lora_fp32.onnx"] != FOUNDATION["foundation_lora_fp32.onnx"]:
        raise ValueError("pinned foundation mismatch")
    return literal_reply(value, count, package["keccak256"].removeprefix("0x"))


def fetch(url, path, expected, ceiling=1 << 30):
    import re
    if not isinstance(url, str) or not url.startswith("https://") or not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError("immutable HTTPS bytes and external hash required")
    allowed = (FOUNDATION_RELEASE, "https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + KIT,
               "https://huggingface.co/answerdotai/ModernBERT-base/resolve/8949b909ec900327062f0ebf497f51aef5e6f0c8/",
               "https://raw.githubusercontent.com/" + REPO + "/")
    if not url.startswith(allowed):
        raise ValueError("immutable public source URL outside allowlist")
    if type(ceiling) is not int or ceiling < 1:
        raise ValueError("positive download byte ceiling required")
    path = Path(path)
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("symlink download/cache path refused")
    try:
        info = path.lstat()
    except FileNotFoundError:
        info = None
    if info is not None:
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("regular cached artifact required")
        if info.st_size <= ceiling:
            h = hashlib.sha256()
            cache_total, cache_started = 0, time.monotonic()
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            except OSError as error:
                raise ValueError("cached artifact changed during verification") from error
            with os.fdopen(fd, "rb") as stream:
                opened = os.fstat(stream.fileno())
                if not stat.S_ISREG(opened.st_mode) or opened.st_size > ceiling:
                    raise ValueError("regular bounded cached artifact required")
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    cache_total += len(chunk)
                    if cache_total > ceiling or time.monotonic() - cache_started > 600:
                        raise ValueError("cached artifact grew beyond its byte/time ceiling")
                    h.update(chunk)
            if h.hexdigest() == expected:
                return
    partial = None
    digest, total = hashlib.sha256(), 0
    started = time.monotonic()
    try:
        # Create an exclusive random sibling, never open a caller-predictable
        # .partial path. Existing failed downloads and symlinks stay untouched.
        fd, name = tempfile.mkstemp(prefix=path.name + ".download-", dir=path.parent)
        partial = Path(name)
        with os.fdopen(fd, "wb") as target, urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "WEB_SEARCH-owner/1"}), timeout=45) as source:
            while chunk := source.read(1 << 20):
                total += len(chunk)
                if total > ceiling or time.monotonic() - started > 600:
                    raise ValueError("download byte or time ceiling")
                target.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("immutable source hash mismatch")
        partial.replace(path)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)


def capture_class(root):
    from common import check_classes, now, public_json
    snapshot = {"schema": 1, "fetched_at": now().isoformat()}
    for name, url in (("family", NODE + "/api/wasm?resolve=" + FAMILY),
                      ("child", NODE + "/engine/v1/intents/" + FAMILY + "_GENERAL")):
        _, raw = public_json(url, 2 << 20)
        snapshot[name] = {"url": url, "raw": raw.decode()}
    check_classes(snapshot, {"family": FAMILY, "child": FAMILY + "_GENERAL", "class": "judgment"})
    write(root / "class-snapshot.json", snapshot)


def prepare(root, training=True):
    remote_only()
    root.mkdir(parents=True, exist_ok=True)
    capture_class(root)  # exact live class before downloading any neural weights
    pin = strict_json(bounded(HERE / "model-sources.json"))
    archive = root / "kit.tar.gz"
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + KIT,
          archive, KIT_SHA, 32 << 20)
    with tarfile.open(archive) as source:
        members = source.getmembers()
        if len(members) > 5000 or sum(m.size for m in members) > 128 << 20:
            raise ValueError("reference source inventory ceiling")
        source.extractall(root, filter="data")
    kit = root / ("telegraph-wasm-baseline-" + KIT)
    trainer = kit / "float_runtime/train_adapter.py"
    expected = strict_json(bounded(HERE / "source-pins.json"))["sources"]["trainer.py"]["sha256"]
    if sha(trainer.read_bytes()) != expected:
        raise ValueError("actual primary trainer source changed")
    for executable, source in (("adapterexec", "./cmd/adapterexec"), ("stagecheck", str(HERE / "stagecheck.go"))):
        subprocess.run(["go", "build", "-mod=readonly", "-trimpath", "-o", str(root / executable), source],
                       cwd=kit, check=True, timeout=600)
    subprocess.run(["go", "test", "-mod=readonly", "-count=1", str(HERE / "stagecheck.go"), str(HERE / "stagecheck_test.go")],
                   cwd=kit, check=True, timeout=180)
    subprocess.run(["go", "test", "-mod=readonly", "-count=1", str(HERE / "stagecheck.go"), str(HERE / "wirecheck_test.go")],
                   cwd=kit, check=True, timeout=180)
    foundation, base = root / "foundation", root / "base"
    foundation.mkdir(exist_ok=True)
    base.mkdir(exist_ok=True)
    for name, expected in FOUNDATION.items():
        fetch(FOUNDATION_RELEASE + name, foundation / name, expected)
    from common import BASE_CONFIG_URL, BASE_CONFIG_SHA
    fetch(BASE_CONFIG_URL, base / "config.json", BASE_CONFIG_SHA, 1 << 20)
    config = strict_json((base / "config.json").read_bytes())
    h, intermediate = config["hidden_size"], config["intermediate_size"]
    write(root / "foundation.json", {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
          "foundation_version": "modernbert-base-lora16-v1", "tokenizer_hash": FOUNDATION["tokenizer.json"],
          "r_max": 16, "num_layers": config["num_hidden_layers"], "hidden_size": h,
          "targets": {"attn.Wqkv": [h, 3 * h], "attn.Wo": [h, h], "mlp.Wi": [h, 2 * intermediate], "mlp.Wo": [intermediate, h]}})
    if training:
        for name, expected in pin["base"]["files"].items():
            fetch(f"https://huggingface.co/{pin['base']['id']}/resolve/{pin['base']['revision']}/{name}", base / name, expected)
        (base / "tokenizer.json").write_bytes((foundation / "tokenizer.json").read_bytes())
    return kit, base, foundation


def parse_package(root, package):
    capture_class(root)
    result = subprocess.run([str(root / "stagecheck"), "--intent", FAMILY, "--child", FAMILY + "_GENERAL",
          "--expected-class", "judgment", "--package", str(package), "--sha256", sha(package.read_bytes()),
          "--foundation", str(root / "foundation.json"), "--foundation-sha256", sha((root / "foundation.json").read_bytes()),
          "--class-snapshot", str(root / "class-snapshot.json"), "--snapshot-sha256", sha((root / "class-snapshot.json").read_bytes()),
          "--out", str(root)], check=True, capture_output=True, timeout=60)
    manifest = strict_json(result.stdout)
    if manifest["manifest"]["verification"] != {"stage1": True, "loaded": False}:
        raise ValueError("full primary parser did not run")
    return manifest


@contextmanager
def engine(root, package, manifest):
    remote_only()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {"PATH": os.environ["PATH"], "HOME": str(root), "FOUNDATION_POOL_SIZE": "1",
           "FOUNDATION_MODEL_PATH": str(root / "foundation/foundation_lora_fp32.onnx"),
           "FOUNDATION_MODEL_HASH": FOUNDATION["foundation_lora_fp32.onnx"],
           "FOUNDATION_ENGINE_WASM_PATH": str(root / "foundation/foundation_engine.wasm"),
           "FOUNDATION_ENGINE_WASM_HASH": FOUNDATION["foundation_engine.wasm"]}
    row = {"keccak": manifest["keccak256"].removeprefix("0x"), "runnable_url": "https://tga1.zkasuran.dev/_wire/runnable"}
    proxy, worker = None, None
    with (root / "reference-runtime.log").open("a") as log:
        process = subprocess.Popen([str(root / "adapterexec"), "--package", str(package), "--listen", f"127.0.0.1:{port}"],
                                   env=env, stdout=log, stderr=log)
        try:
            ready = time.monotonic() + 180
            while time.monotonic() < ready:
                if process.poll() is not None:
                    raise RuntimeError("reference foundation load failed")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    time.sleep(.5)
            else:
                raise TimeoutError("reference foundation load ceiling")
            backend = Backend(port, timeout=180)
            proxy = BoundedServer(("127.0.0.1", 0), row, backend)
            worker = threading.Thread(target=proxy.serve_forever, daemon=True)
            worker.start()
            def request(vectors, override=None):
                body = {"batch": [{"tokens": ids} for ids in vectors], **identity(row)}
                body["input_root"] = input_root(body["batch"])
                body.update(override or {})
                message = urllib.request.Request(f"http://127.0.0.1:{proxy.server_port}/runnable", data=json.dumps(body, separators=(",", ":")).encode(),
                                                 headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(message, timeout=600) as response:
                    raw = response.read((1 << 20) + 1)
                if len(raw) > 1 << 20:
                    raise ValueError("literal reply ceiling")
                return literal_reply(strict_json(raw), len(vectors), row["keccak"], body["input_root"])
            yield request, row
        finally:
            if proxy is not None:
                proxy.shutdown()
                proxy.server_close()
            if worker is not None:
                worker.join(timeout=5)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def sequence_scores(request, vectors):
    return [score for start in range(0, len(vectors), 4) for score in request(vectors[start:start + 4])]


def source_token_gate(kit, foundation, splits):
    # Source-only gate, called remotely before any model library is imported.
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    vectors = strict_json((kit / "float_runtime/testdata/tokenizer_vectors.json").read_bytes())["vectors"]
    for vector in vectors:
        if tokenizer.encode(vector["text"], add_special_tokens=vector["add_special_tokens"]).ids != vector["ids"]:
            raise ValueError("actual published tokenizer conformance failure")
    spec = importlib.util.spec_from_file_location("owner_source_trainer", kit / "float_runtime/train_adapter.py")
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    gate = trainer.gate_dataset(splits["train"], trainer.Hyperparams())
    if not gate["ok"] or gate["metrics"]["surface_probe_acc"] is None:
        raise ValueError("unchanged primary dataset gate failed")
    checked = 0
    for rows in splits.values():
        for row in rows:
            ids = [tokenizer.encode(trainer.render(row["prompt"], row[side])).ids for side in ("chosen", "rejected")]
            if ids[0] == ids[1] or max(map(len, ids)) > 256 or trainer.render(row["prompt"], row["chosen"]) != render(row["prompt"], row["chosen"]):
                raise ValueError("complete decisive source and answer not visible")
            checked += 1
    return tokenizer, trainer, {"pairs": checked, "conformance_vectors": len(vectors), "upstream": gate}


def train(root):
    remote_only()
    started = time.monotonic()
    recipe, _, splits = read_splits()
    kit, base, foundation = prepare(root)
    tokenizer, trainer, data_gate = source_token_gate(kit, foundation, splits)
    write(root / "data-gate.json", data_gate)
    sys.path.insert(0, str(kit))
    import numpy as np
    import torch
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import metadata, tensor_names, write_package
    from torch.utils.data import DataLoader
    from transformers import AutoConfig
    if torch.cuda.is_available():
        raise RuntimeError("standard public CPU only")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    hp = trainer.Hyperparams(**recipe["hyperparameters"], grad_checkpointing=False, early_stopping=False, gate_min_acc=.8)
    model, tok = trainer.build_model(str(base), hp)
    for rows in splits.values():
        for row in rows:
            row["margin"] = 0.0
            for side in ("chosen", "rejected"):
                text = render(row["prompt"], row[side])
                if tok(text, truncation=False)["input_ids"] != tokenizer.encode(text).ids:
                    raise ValueError("actual train/node token IDs differ")
    cfg = AutoConfig.from_pretrained(base, local_files_only=True)
    def state_hash():
        digest = hashlib.sha256()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                digest.update(name.encode())
                digest.update(parameter.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
    def save(destination):
        destination.mkdir(exist_ok=True)
        model.save_pretrained(destination)
        trainer.save_reward_head(model, destination)
        head = torch.load(destination / "reward_head.pt", weights_only=True, map_location="cpu")
        for path in destination.glob("*.safetensors"):
            preserve_artifact(path)
        preserve_artifact(destination / "reward_head.pt")
        restored = torch.load(destination / "reward_head.pt", weights_only=True, map_location="cpu")
        if head.keys() != restored.keys() or any(not torch.equal(head[key], restored[key]) for key in head):
            raise ValueError("inert upstream terms changed checkpoint tensors")
        tok.save_pretrained(destination)
        (destination / "tokenizer.json").write_bytes((foundation / "tokenizer.json").read_bytes())
    def evaluate(rows):
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        training = model.training
        try:
            with torch.random.fork_rng(devices=[]):
                return trainer.evaluate(model, tok, rows, hp)
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            model.train(training)
    def export(directory, filename, validation):
        tensors, parameters = read_peft(directory, cfg)
        meta = metadata(cls="judgment", intent=FAMILY, foundation_hash=FOUNDATION["foundation_lora_fp32.onnx"],
              foundation_version="modernbert-base-lora16-v1", rank=parameters["rank"], alpha=parameters["alpha"],
              target_modules=parameters["target_modules"], max_tokens=256,
              training={"family": FAMILY, "children": [{"intent": FAMILY + "_GENERAL", "dataset_version": recipe["files"]["validation"]["sha256"],
                         "pairs": 64, "accuracy": validation["pairwise_accuracy"], "held_out": True}]})
        meta.update(licence_metadata())
        meta["x-source-manifest"] = recipe["source_manifest_sha256"]
        meta["x-authorship"] = "Source metadata retained; preference queries and version swaps constructed by AI-assisted code, not human-reviewed."
        package = root / filename
        write_package(package, meta, {name: tensors[name] for name in tensor_names(cfg.num_hidden_layers)})
        return package, parse_package(root, package)
    initial_state = state_hash()
    initial_validation = evaluate(splits["validation"])
    save(root / "initial")
    initial_package, initial_manifest = export(root / "initial", "UNTRAINED.safetensors", initial_validation)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=hp.lr, weight_decay=hp.weight_decay)
    sampler = torch.Generator(device="cpu").manual_seed(42)
    loader = DataLoader(splits["train"], batch_size=1, shuffle=True, generator=sampler,
                        collate_fn=trainer.PairCollator(tok, 256), num_workers=0)
    if len(loader) % 16:
        raise ValueError("fixed complete accumulation groups required")
    best, history, optimizer_steps, times = None, [], 0, []
    for epoch in range(1, hp.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_started = time.monotonic()
        for i, batch in enumerate(loader, 1):
            chosen = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rejected = trainer._reward(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
            loss = trainer.bt_loss(chosen, rejected, batch["margin"])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            (loss / 16).backward()
            if i % 16 == 0:
                torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                times.append(time.monotonic() - step_started)
                save(root / "checkpoint")
                write(root / "checkpoint.json", {"phase": "training", "epoch": epoch, "steps": optimizer_steps,
                      "elapsed_seconds": time.monotonic() - started, "step_seconds": times, "audit_scored": False,
                      "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"], "initial_state_sha256": initial_state})
                remaining = (hp.epochs * len(loader) / 16 - optimizer_steps) * max(times)
                if time.monotonic() - started + remaining > 270 * 60:
                    raise TimeoutError("measured training cannot fit bounded runtime/audit reserve")
                step_started = time.monotonic()
        validation = evaluate(splits["validation"])
        history.append({"epoch": epoch, "validation": validation})
        if best is None or validation["val_loss"] < best["val_loss"]:
            best = {"epoch": epoch, "val_loss": validation["val_loss"], "state_sha256": state_hash()}
            save(root / "selected")
        write(root / "development-selection.json", {"best": best, "history": history, "audit_scored": False})
    # Load only the saved development-selected checkpoint, before reading audit scores.
    selected = root / "selected"
    validation = history[best["epoch"] - 1]["validation"]
    package, manifest = export(selected, "WEB_SEARCH.safetensors", validation)
    frozen_sha = sha(package.read_bytes())
    write(root / "frozen-candidate.json", {"package_sha256": frozen_sha, "selected": best,
          "initial_package_sha256": sha(initial_package.read_bytes()), "audit_scored": False})
    model = None
    del parameters, optimizer, loader, sampler, chosen, rejected, loss
    gc.collect()
    vectors = [tokenizer.encode(render(row["prompt"], row[side])).ids for row in splits["validation"] for side in ("chosen", "rejected")]
    with engine(root, package, manifest) as (request, row):
        validation_bits = sequence_scores(request, vectors)
        fixed = request(vectors[:8])
        repeated = request(vectors[:8])
        refusals = []
        for key in ("foundation_hash", "tokenizer_hash", "adapter_hash"):
            try:
                request(vectors[:1], {key: "0" * 64})
            except urllib.error.HTTPError as error:
                if error.code != 400:
                    raise
                refusals.append(key)
            else:
                raise ValueError("actual identity refusal control failed")
        for invalid in ([], [-1]):
            try:
                request([invalid])
            except urllib.error.HTTPError as error:
                if error.code != 400:
                    raise
            else:
                raise ValueError("actual token refusal failed")
    with engine(root, package, manifest) as (request, _):
        fresh = request(vectors[:8])
    if fixed != repeated or fixed != fresh:
        raise ValueError("actual fresh reference-engine bits changed")
    with engine(root, initial_package, initial_manifest) as (request, _):
        initial_validation_bits = sequence_scores(request, vectors)
    trained_validation = metrics(splits["validation"], validation_bits)
    untrained_validation = metrics(splits["validation"], initial_validation_bits)
    validation_controls = shortcuts(splits["validation"])
    write(root / "validation-reference.json", {"trained": trained_validation, "untrained": untrained_validation,
          "shortcuts": validation_controls, "fixed_vectors": [{"tokens": tokens, "score_bits": value} for tokens, value in zip(vectors[:8], fixed)]})
    if not quality_gate(trained_validation, untrained_validation["accuracy"], max(validation_controls.values())):
        write(root / "result.json", {"status": "FAILED_VALIDATION", "validation": trained_validation, "audit_scored": False})
        return 1
    # This marker is written before the independent audit. A retry cannot draw it again.
    marker = root / "audit-started.json"
    if marker.exists():
        raise ValueError("independent audit already started; preserve its exact result")
    write(marker, {"frozen_package_sha256": frozen_sha, "audit_sha256": recipe["files"]["audit"]["sha256"]})
    audit_vectors = [tokenizer.encode(render(row["prompt"], row[side])).ids for row in splits["audit"] for side in ("chosen", "rejected")]
    with engine(root, package, manifest) as (request, _):
        trained_bits = sequence_scores(request, audit_vectors)
    with engine(root, initial_package, initial_manifest) as (request, _):
        untrained_bits = sequence_scores(request, audit_vectors)
    audit = metrics(splits["audit"], trained_bits)
    untrained = metrics(splits["audit"], untrained_bits)
    controls = shortcuts(splits["audit"])
    from scorer import score as lookup
    lookup_accuracy = sum(lookup(row["prompt"], row["chosen"]) > lookup(row["prompt"], row["rejected"]) for row in splits["audit"]) / len(splits["audit"])
    audit.update(untrained_accuracy=untrained["accuracy"], shortcut_accuracy=max(controls.values()), shortcuts=controls,
                 task_specific_lookup_accuracy=lookup_accuracy, lookup_limit="Evidence binding proxy, not universal factual proof")
    passed = quality_gate(audit, untrained["accuracy"], max(controls.values()))
    write(root / "audit.json", {"status": "PASS" if passed else "FAIL", "metrics": audit,
          "package_sha256": frozen_sha, "candidate_score_bits": trained_bits, "untrained_score_bits": untrained_bits,
          "audit_used_for_selection": False, "independent_groups": audit["independent_pages"]})
    if not passed:
        write(root / "result.json", {"status": "FAILED_AUDIT", "audit": audit, "package_sha256": frozen_sha})
        return 1
    if sha(package.read_bytes()) != frozen_sha:
        raise ValueError("frozen candidate bytes changed after audit")
    ready = {"schema_version": 1, "status": "REGISTRATION_READY", "family": FAMILY,
             "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
             "package_sha256": frozen_sha, "package_keccak": manifest["keccak256"].removeprefix("0x"),
             "package_bytes": package.stat().st_size, "foundation_sha256": FOUNDATION["foundation_lora_fp32.onnx"],
             "engine_sha256": FOUNDATION["foundation_engine.wasm"], "tokenizer_sha256": FOUNDATION["tokenizer.json"],
             "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True,
                        "source_manifest_sha256": recipe["source_manifest_sha256"], "independent_groups": audit["independent_pages"]},
             "quality": {"passed": True, "audit_pairs": len(splits["audit"]), "all_supported_formats_tested": True, "formats": {"raw_json": audit}},
             "runtime": {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
                         "negative_identity_controls": len(refusals), "vectors": [{"tokens": ids, "score_bits": value} for ids, value in zip(vectors[:8], fixed)]},
             "serving_commit": os.environ["GITHUB_SHA"], "node": NODE, "owner_attempt": recipe["owner_attempt"],
             "package_manifest": {key: manifest[key] for key in ("sha256", "keccak256", "bytes")}, "recipe_sha256": sha((HERE / "recipe.json").read_bytes()),
             "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "tokenizers", "numpy")}}
    ready_report(ready)
    write(root / "registration-ready.json", ready)
    write(root / "result.json", {"status": "REGISTRATION_READY", "report_sha256": sha((root / "registration-ready.json").read_bytes())})
    return 0


if __name__ == "__main__":
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    def stop(signum, frame):
        raise TimeoutError("bounded public CPU owner job ended")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(305 * 60)
    sys.exit(train(args.root.resolve()))
