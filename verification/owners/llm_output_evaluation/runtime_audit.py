# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Once-only source-held-out audit in the primary native and WASM runtimes."""
import gc
import json
import math
import os
import random
import socket
import struct
import subprocess
import time
import urllib.request
import threading
import urllib.error

from data import FAMILY, RATINGS, FORMATS, format_pair, control_scores, load_splits, render
from source_gate import digest, write
from proxy import Backend, BoundedServer, input_root, identity as wire_identity
from token_contract import checked_tokens
from presentation import evaluation_view
from training_recipe import pair_bytes


def decode(bits):
    if type(bits) is not int or not 0 <= bits <= 0xffffffff:
        raise ValueError("invalid literal uint32 FP32 wire score")
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    if not math.isfinite(value):
        raise ValueError("nonfinite wire score")
    return value


def metrics(rows, values):
    if len(values) != 2 * len(rows) or not rows:
        raise ValueError("audit score count mismatch")
    if any(not isinstance(row, dict) or type(row.get("label")) is not str
           or row["label"] not in RATINGS for row in rows):
        raise ValueError("audit contains an unknown rating label")
    if {row["label"] for row in rows} != set(RATINGS):
        raise ValueError("audit requires every supported rating label")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("audit requires finite numeric scores")
    wins = [values[2*i] > values[2*i+1] for i in range(len(rows))]
    slices = {label: [w for r, w in zip(rows, wins) if r["label"] == label] for label in RATINGS}
    return {"originals": len(rows), "correct": sum(wins), "accuracy": sum(wins) / len(rows),
            "rating_slices": {k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in slices.items()}}


def run_audit(root, reference, identity):
    if os.environ.get("GITHUB_RUN_ATTEMPT") != "1":
        raise ValueError("audit reruns are forbidden on the same frozen sample")
    marker = root / "audit-started.json"
    if marker.exists():
        raise ValueError("independent audit is once-only; no retry selection on this sample")
    selected = json.loads((root / "selected.json").read_text())
    if (selected["github_commit"] != os.environ["GITHUB_SHA"]
            or selected["github_run_id"] != os.environ["GITHUB_RUN_ID"]):
        raise ValueError("audit run/commit differs from the frozen training identity")
    package = root / (FAMILY + ".safetensors")
    if digest(package.read_bytes()) != selected["package_sha256"] or selected["audit_scored"] is not False:
        raise ValueError("audit candidate identity changed")
    if digest((root / "data/frozen-baseline.json").read_bytes()) != selected["baseline_sha256"]:
        raise ValueError("train-fitted baseline bytes changed")
    recipe_path = root / "training-recipe.json"
    if digest(recipe_path.read_bytes()) != selected["training_recipe_sha256"]:
        raise ValueError("selected training recipe identity changed")
    recipe = json.loads(recipe_path.read_bytes())
    if (digest((root / "data/training_views.jsonl").read_bytes()) != recipe["training_views_sha256"]
            or digest((root / "data/train.jsonl").read_bytes()) != recipe["training_source_sha256"]
            or selected["baseline_sha256"] != recipe["baseline_sha256"]):
        raise ValueError("frozen training-view/source/baseline identity changed")
    report, splits = load_splits(root / "data")
    if report["files"] != selected["data"]["files"]:
        raise ValueError("audit data identity changed")
    # Exclusive marker refuses concurrent audit processes in the same artifact root.
    with marker.open("x") as stream:
        json.dump({"package_sha256": selected["package_sha256"], "audit_sha256": report["files"]["audit"]["sha256"]}, stream)
    helper, foundation, base, kit = reference(root)
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from float_runtime import train_adapter as trainer
    import importlib.metadata
    if torch.cuda.is_available():
        raise RuntimeError("CPU audit refuses GPU")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    def seed():
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
    seed()
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    trainer.check_tokenizer(tok)
    from tokenizers import Tokenizer
    node = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    node.no_padding()
    node.no_truncation()
    format_rows = {name: [evaluation_view(r, name, recipe["recipe"]) for r in splits["audit"]] for name in FORMATS}
    rows = [r for name in FORMATS for r in format_rows[name]]
    rendered_views_sha256 = digest(pair_bytes(rows))
    if len(splits["audit"]) < 64 or len({r["group_id"] for r in splits["audit"]}) < 32:
        raise ValueError("complete independent 64-pair audit required")
    texts = [render(r["prompt"], r[s]) for r in rows for s in ("chosen", "rejected")]
    vectors = [checked_tokens(text, tok, node, 256) for text in texts]
    backbone = AutoModelForSequenceClassification.from_pretrained(base, local_files_only=True, num_labels=1,
                         torch_dtype=torch.float32, attn_implementation="eager")
    model = PeftModel.from_pretrained(backbone, root / "adapter", local_files_only=True)
    model.eval()
    native = []
    with torch.no_grad():
        for text in texts:
            encoded = tok(text, return_tensors="pt", truncation=False)
            native.append(trainer._reward(model, encoded["input_ids"], encoded["attention_mask"]).item())
    if any(not math.isfinite(v) for v in native):
        raise ValueError("nonfinite native audit")
    del model, backbone
    gc.collect()
    seed()
    initial, _ = trainer.build_model(str(base), trainer.Hyperparams(max_length=256, lora_rank=8, grad_checkpointing=False))
    initial_metrics = {name: trainer.evaluate(initial, tok, rs, trainer.Hyperparams(max_length=256, eval_batch=1)) for name, rs in format_rows.items()}
    del initial
    gc.collect()
    baseline = json.loads((root / "data/frozen-baseline.json").read_text())
    controls = {name: control_scores(rs, baseline) for name, rs in format_rows.items()}
    executable = root / "adapterexec"
    subprocess.run(["go", "build", "-mod=readonly", "-trimpath", "-o", str(executable), "./cmd/adapterexec"],
                   cwd=kit, check=True, timeout=600)
    expected = json.loads((root / "package-check.json").read_text())["keccak256"].removeprefix("0x")
    parser = json.loads((root / "package-check.json").read_text())
    if parser.get("manifest", {}).get("verification", {}).get("stage1") is not True:
        raise ValueError("full primary parser was not observed passing")
    env = {**os.environ, "FOUNDATION_MODEL_PATH": str(foundation / "foundation_lora_fp32.onnx"),
           "FOUNDATION_MODEL_HASH": helper.FOUNDATION["foundation_lora_fp32.onnx"],
           "FOUNDATION_ENGINE_WASM_PATH": str(foundation / "foundation_engine.wasm"),
           "FOUNDATION_ENGINE_WASM_HASH": helper.FOUNDATION["foundation_engine.wasm"]}
    def execute(vs, name):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with (root / (name + ".log")).open("w") as log:
            server = subprocess.Popen([str(executable), "--package", str(package), "--listen", f"127.0.0.1:{port}"],
                                      env=env, stdout=log, stderr=log)
            proxy = None
            thread = None
            try:
                ready = time.monotonic() + 180
                while True:
                    if server.poll() is not None or time.monotonic() > ready:
                        raise RuntimeError("reference executor readiness failed")
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=1):
                            break
                    except OSError:
                        time.sleep(1)
                row = {"keccak": expected, "runnable_url": "https://tga1.zkasuran.dev/_llm/runnable"}
                proxy = BoundedServer(("127.0.0.1", 0), row, Backend(port))
                thread = threading.Thread(target=proxy.serve_forever, daemon=True)
                thread.start()
                negative = 0
                for key in wire_identity(row):
                    body = {"batch": [{"tokens": [50281,22,50282]}], **wire_identity(row), key: "0" * 64}
                    req = urllib.request.Request(f"http://127.0.0.1:{proxy.server_port}/runnable", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                    try:
                        urllib.request.urlopen(req, timeout=10).close()
                    except urllib.error.HTTPError as error:
                        if error.code != 400 or json.loads(error.read()) != {"error": "request refused"}:
                            raise ValueError("negative identity refusal differs from literal wire contract")
                        negative += 1
                    else:
                        raise ValueError("real public executor proxy accepted a wrong identity")
                results = []
                for start in range(0, len(vs), 4):
                    batch = vs[start:start+4]
                    batch_body = [{"tokens": ids} for ids in batch]
                    req = urllib.request.Request(f"http://127.0.0.1:{proxy.server_port}/runnable",
                          data=json.dumps({"batch": [{"tokens": ids} for ids in batch],
                                "foundation_hash": helper.FOUNDATION["foundation_lora_fp32.onnx"],
                                "adapter_hash": expected, "input_root": input_root(batch_body),
                                "tokenizer_hash": helper.FOUNDATION["tokenizer.json"]}).encode(),
                          headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=180) as response:
                        raw = response.read((1 << 20) + 1)
                    if len(raw) > 1 << 20:
                        raise ValueError("runtime response ceiling")
                    value = json.loads(raw)
                    if (value.get("adapter_hash") != expected
                            or value.get("foundation_hash") != helper.FOUNDATION["foundation_lora_fp32.onnx"]
                            or value.get("tokenizer_hash") != helper.FOUNDATION["tokenizer.json"]
                            or len(value.get("results", [])) != len(batch)):
                        raise ValueError("literal runtime identity/count mismatch")
                    for item in value["results"]:
                        if "error" in item:
                            raise ValueError("runtime item failed")
                        decode(item["score"])
                        results.append(item["score"])
                return results, negative
            finally:
                if proxy is not None:
                    proxy.shutdown()
                    proxy.server_close()
                    thread.join(timeout=10)
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10)
    # Same scoring audit, repeat and fresh process reproduction, not another selected candidate.
    bits, negative = execute(vectors + vectors[:8], "runtime-primary")
    first, repeated = bits[:len(vectors)], bits[len(vectors):]
    fresh, fresh_negative = execute(vectors[:8], "runtime-fresh")
    values = [decode(b) for b in first]
    width = len(splits["audit"]) * 2
    actual = {name: metrics(rs, values[i*width:(i+1)*width]) for i, (name, rs) in enumerate(format_rows.items())}
    agreement = all(abs(a-b) <= 0.001 + 0.001 * abs(a) for a, b in zip(native, values))
    same_order = all((native[i] > native[i+1]) == (values[i] > values[i+1]) for i in range(0, len(native), 2))
    format_quality = {name: {"accuracy": actual[name]["accuracy"],
             "per_label": {k: v["accuracy"] for k, v in actual[name]["rating_slices"].items()},
             "per_label_counts": {k: v["n"] for k, v in actual[name]["rating_slices"].items()},
             "untrained_accuracy": initial_metrics[name]["pairwise_accuracy"],
             "shortcut_accuracy": max(controls[name].values())} for name in FORMATS}
    passed = (all(v["accuracy"] >= 0.80 and min(v["per_label"].values()) >= 0.70
                  and v["accuracy"] + 1e-9 >= v["untrained_accuracy"] + 0.05
                  and v["accuracy"] + 1e-9 >= v["shortcut_accuracy"] + 0.10
                  and v["shortcut_accuracy"] <= 0.65 for v in format_quality.values())
              and first[:8] == repeated and fresh == first[:8] and agreement and same_order and negative >= 3)
    output = {**identity(root), "outcome": "PASS" if passed else "FAIL", "candidate": selected,
              "runtime": actual, "controls": controls, "untrained": initial_metrics,
              "native_export_tolerance_pass": agreement, "native_export_order_pass": same_order,
              "exact_repeat_bits": first[:8] == repeated, "exact_fresh_process_bits": fresh == first[:8],
              "bits": first, "native_scores": native, "foundation": helper.FOUNDATION,
              "package_sha256": selected["package_sha256"], "runtime_binary_sha256": digest(executable.read_bytes()),
              "tokenizer_inputs_compared": len(vectors), "native_node_token_parity": True,
              "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "numpy", "safetensors")},
              "node_qualification": "not attempted", "quality": format_quality,
              "negative_identity_controls": negative, "limits": ["source-held-out scores are not node qualification", "rating calibration is a proxy"]}
    write(root / "audit-result.json", output)
    if not passed:
        raise ValueError("independent audit quality/reproduction failed, no retry on this sample")
    source_manifest = {"source_pins_sha256": digest((__import__('pathlib').Path(__file__).parent / 'sources.json').read_bytes()),
                        "data_files": report["files"], "source_inventory_sha256": report["source_inventory_sha256"],
                        "training_recipe_sha256": selected["training_recipe_sha256"],
                        "training_views_sha256": recipe["training_views_sha256"],
                        "audit_rendered_views_sha256": rendered_views_sha256,
                       "commit": os.environ["GITHUB_SHA"], "run_id": int(os.environ["GITHUB_RUN_ID"])}
    write(root / "source-runtime-manifest.json", source_manifest)
    ready = {"schema_version": 1, "status": "REGISTRATION_READY", "family": FAMILY,
             "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
             "package_sha256": selected["package_sha256"], "package_keccak": expected,
             "foundation_sha256": helper.FOUNDATION["foundation_lora_fp32.onnx"],
             "engine_sha256": helper.FOUNDATION["foundation_engine.wasm"], "tokenizer_sha256": helper.FOUNDATION["tokenizer.json"],
             "source": {"normalized_overlap": 0, "audit_used_for_selection": False,
                        "grants_verified": True, "independent_groups": len({r["group_id"] for r in splits["audit"]}),
                        "source_manifest_sha256": digest((root / "source-runtime-manifest.json").read_bytes())},
             "quality": {"passed": True, "audit_pairs": len(splits["audit"]), "all_supported_formats_tested": True, "formats": format_quality},
             "runtime": {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
                         "negative_identity_controls": negative, "vectors": [{"tokens": ids, "score_bits": b} for ids, b in zip(vectors[:8], first[:8])]},
             "serving_commit": os.environ["GITHUB_SHA"], "owner_attempt": os.environ["OWNER_ATTEMPT"],
             "evidence": {"audit_result_sha256": digest((root / "audit-result.json").read_bytes()),
                          "runtime_binary_sha256": digest(executable.read_bytes())}}
    write(root / "registration-ready.json", ready)
