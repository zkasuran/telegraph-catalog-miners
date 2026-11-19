# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only exported scoring of all three views of the same 256 originals."""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path

from data import decode_jsonl, digest, pair, write_json
from runtime_check import decode_bits
from sentiment_recover import (
    FAMILY,
    FORMATS,
    FOUNDATION,
    LABELS,
    PACKAGE_SHA,
    original_report,
    remote_only,
    verify_data,
)


def format_rows(originals):
    if len(originals) != 256 or Counter(r["label"] for r in originals) != {"Negative": 128, "Positive": 128}:
        raise ValueError("all-format runtime requires exactly 256 balanced originals")
    if len({r["group_id"] for r in originals}) != 256:
        raise ValueError("duplicate held-out original")
    return {name: [{**pair(FAMILY, {**row, "text": json.loads(row["prompt"])["text"]}, style),
                    "margin": 0.0} for row in originals]
            for style, name in enumerate(FORMATS)}


def check_response(data, count, manifest):
    if (data.get("foundation_hash") != FOUNDATION["foundation_lora_fp32.onnx"] or
            data.get("tokenizer_hash") != FOUNDATION["tokenizer.json"] or
            not isinstance(data.get("adapter_hash"), str) or
            data["adapter_hash"].removeprefix("0x") != manifest["keccak256"].removeprefix("0x")):
        raise ValueError("runtime identity mismatch")
    if not isinstance(data.get("results"), list) or len(data["results"]) != count:
        raise ValueError("runtime returned wrong result count")
    bits = [r["score"] for r in data["results"]]
    for value in bits:
        decode_bits(value)
    return bits


def metrics(rows, bits, repeated):
    if len(bits) != 2 * len(rows) or len(repeated) != len(bits):
        raise ValueError("missing format scores or repeated scores")
    # Check both streams before comparing so NaN cannot hide behind a bit match.
    values = [decode_bits(b) for b in bits]
    for b in repeated:
        decode_bits(b)
    if bits != repeated:
        raise ValueError("runtime score bits changed on identical inputs")
    wins = [values[2*i] > values[2*i+1] for i in range(len(rows))]
    counts = Counter(r["label"] for r in rows)
    if set(counts) != LABELS or counts != {"Negative": 128, "Positive": 128}:
        raise ValueError("missing runtime label or original")
    correct = {label: sum(w for row, w in zip(rows, wins) if row["label"] == label) for label in sorted(LABELS)}
    length = sum(0.5 if len(r["chosen"]) == len(r["rejected"]) else
                 float(len(r["chosen"]) > len(r["rejected"])) for r in rows) / len(rows)
    return {"n_eval": len(rows), "correct": sum(wins), "pairwise_accuracy": sum(wins) / len(rows),
            "accuracy_by_label": {label: correct[label] / counts[label] for label in sorted(LABELS)},
            "counts_by_label": {label: {"correct": correct[label], "n": counts[label]} for label in sorted(LABELS)},
            "length_baseline": length, "repeat_bit_vectors": len(bits)}


def heldout_verdict(formats, report):
    failures = []
    if set(formats) != set(FORMATS):
        failures.append("missing held-out response format")
    for name in FORMATS:
        if name not in formats:
            continue
        m = formats[name]
        values = [m["pairwise_accuracy"], m["length_baseline"], *m["accuracy_by_label"].values()]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in values):
            failures.append(name + " invalid or nonfinite quality score")
            continue
        initial = report["untrained"][name]
        # Recover the unique exact count from the original four-decimal metric.
        counts = [n for n in range(257) if round(n / 256, 4) == initial["pairwise_accuracy"]]
        if len(counts) != 1:
            raise ValueError("ambiguous original seed-matched baseline")
        m["original_untrained_accuracy"] = counts[0] / 256
        m["untrained_baseline_scope"] = "original PyTorch run, not replayed in recovery"
        m["beats_length_baseline"] = m["pairwise_accuracy"] >= m["length_baseline"] + 0.10
        if (m["n_eval"] != 256 or set(m["accuracy_by_label"]) != LABELS or
                m["pairwise_accuracy"] < 0.80 or
                any(v < 0.70 for v in m["accuracy_by_label"].values()) or
                not m["beats_length_baseline"] or
                m["pairwise_accuracy"] < m["original_untrained_accuracy"] + 0.05 or
                m["repeat_bit_vectors"] != 512):
            failures.append(name + " source-held-out runtime gate failed")
    return {"verdict": "FAIL" if failures else "PASS", "failures": failures}


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    root = parser.parse_args().root.resolve()
    report = original_report(root)
    verify_data(root)
    manifest = json.loads((root / "package-manifest.json").read_bytes())
    if manifest["sha256"] != PACKAGE_SHA or manifest["manifest"]["verification"] != {"stage1": True, "loaded": False}:
        raise ValueError("runtime requires unchanged Stage 1 package")
    foundation = root / "foundation"
    for name, expected in FOUNDATION.items():
        # Stream large model files without creating a second in-memory copy.
        h = hashlib.sha256()
        with (foundation / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != expected:
            raise ValueError("runtime input hash mismatch: " + name)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    views = format_rows(decode_jsonl((root / "data/test.jsonl").read_bytes()))
    vectors = {}
    for name, rows in views.items():
        vectors[name] = []
        for row in rows:
            for side in ("chosen", "rejected"):
                text = f"### Task:\n{row['prompt']}\n\n### Answer:\n{row[side]}"
                ids = tokenizer.encode(text).ids
                if not ids or len(ids) > 256:
                    raise ValueError("source pair exceeds runtime token ceiling")
                vectors[name].append(ids)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {**os.environ, "FOUNDATION_MODEL_PATH": str(foundation / "foundation_lora_fp32.onnx"),
           "FOUNDATION_MODEL_HASH": FOUNDATION["foundation_lora_fp32.onnx"],
           "FOUNDATION_ENGINE_WASM_PATH": str(foundation / "foundation_engine.wasm"),
           "FOUNDATION_ENGINE_WASM_HASH": FOUNDATION["foundation_engine.wasm"]}
    env.pop("GITHUB_TOKEN", None)
    started = time.monotonic()
    deadline = started + 7200
    scores, formats = {}, {}
    with (root / "runtime.log").open("w") as log:
        server = subprocess.Popen([str(root / "adapterexec"), "--package", str(root / (FAMILY + ".safetensors")),
                                   "--listen", f"127.0.0.1:{port}"], env=env, stdout=log, stderr=log)
        try:
            ready = time.monotonic() + 180
            while time.monotonic() < ready:
                if server.poll() is not None:
                    raise RuntimeError("reference executor failed to load")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    time.sleep(1)
            else:
                raise RuntimeError("reference executor readiness timeout")

            def score(batch):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("all-format runtime time ceiling exceeded")
                if (root / "runtime.log").stat().st_size > 16 << 20:
                    raise RuntimeError("runtime diagnostic byte ceiling exceeded")
                request = urllib.request.Request(f"http://127.0.0.1:{port}/runnable",
                          data=json.dumps({"batch": [{"tokens": ids} for ids in batch],
                                           "foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
                                           "tokenizer_hash": FOUNDATION["tokenizer.json"]}).encode(),
                          headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=min(180, remaining)) as response:
                    raw = response.read((1 << 20) + 1)
                    if response.status != 200 or len(raw) > 1 << 20:
                        raise ValueError("runtime response status or byte ceiling")
                return check_response(json.loads(raw), len(batch), manifest)

            for name in FORMATS:
                streams = []
                for _ in range(2):
                    bits = []
                    for start in range(0, 512, 8):
                        bits.extend(score(vectors[name][start:start + 8]))
                    streams.append(bits)
                scores[name] = {"score_bits": streams[0], "repeated_score_bits": streams[1],
                                "token_vectors_sha256": digest(json.dumps(vectors[name]).encode())}
                formats[name] = metrics(views[name], *streams)
                write_json(root / "runtime-score-bits.json", scores)
            verdict = heldout_verdict(formats, report)
            evidence = {"family": FAMILY, **verdict, "originals": 256, "format_views": 768,
                        "formats": formats, "adapter_hash": manifest["keccak256"],
                        "package_sha256": PACKAGE_SHA, "foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
                        "tokenizer_hash": FOUNDATION["tokenizer.json"], "engine_hash": FOUNDATION["foundation_engine.wasm"],
                        "repeat_bit_vectors": 1536, "elapsed_seconds": round(time.monotonic() - started, 2),
                        "training_identity": {"run_id": report["run_id"], "commit": report["commit"]},
                        "recovery_identity": {"run_id": os.environ["GITHUB_RUN_ID"], "commit": os.environ["GITHUB_SHA"]},
                        "node_qualification": "not attempted"}
            write_json(root / "runtime-result.json", evidence)
            if verdict["verdict"] != "PASS":
                raise ValueError("exported all-format held-out gate failed")
        except Exception as error:
            write_json(root / "runtime-error.json", {"phase": "exported_runtime", "verdict": "FAIL",
                       "error_type": type(error).__name__, "completed_formats": sorted(formats),
                       "package_sha256": PACKAGE_SHA})
            raise
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)


if __name__ == "__main__":
    main()
