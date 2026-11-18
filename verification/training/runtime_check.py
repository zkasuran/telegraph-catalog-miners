# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Verify exported held-out ordering and repeated bits in the real node runtime."""
import argparse
import json
import math
import os
import socket
import struct
import subprocess
import time
import urllib.request
from pathlib import Path

from data import decode_jsonl, write_json
from run import FOUNDATION, remote_only


def decode_bits(bits):
    if isinstance(bits, bool) or not isinstance(bits, int) or not 0 <= bits <= 0xffffffff:
        raise ValueError("invalid FP32 score bits")
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    if not math.isfinite(value):
        raise ValueError("non-finite runtime score")
    return value


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    root = parser.parse_args().root
    from tokenizers import Tokenizer
    report = json.loads((root / "result.json").read_text())
    manifest = json.loads((root / "package-manifest.json").read_text())
    family = report["family"]
    foundation = root / "foundation"
    tokenizer = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=256)
    rows = decode_jsonl((root / "data" / "test.jsonl").read_bytes())
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {**os.environ, "FOUNDATION_MODEL_PATH": str(foundation / "foundation_lora_fp32.onnx"),
           "FOUNDATION_MODEL_HASH": FOUNDATION["foundation_lora_fp32.onnx"],
           "FOUNDATION_ENGINE_WASM_PATH": str(foundation / "foundation_engine.wasm"),
           "FOUNDATION_ENGINE_WASM_HASH": FOUNDATION["foundation_engine.wasm"]}
    with (root / "runtime.log").open("w") as log:
        server = subprocess.Popen([str(root / "adapterexec"), "--package", str(root / (family + ".safetensors")),
                                   "--listen", f"127.0.0.1:{port}"], env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
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
                request = urllib.request.Request(f"http://127.0.0.1:{port}/runnable",
                         data=json.dumps({"batch": [{"tokens": ids} for ids in batch],
                                          "foundation_hash": env["FOUNDATION_MODEL_HASH"],
                                          "tokenizer_hash": FOUNDATION["tokenizer.json"]}).encode(),
                         headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=180) as response:
                    data = json.loads(response.read(1 << 20))
                if (data.get("foundation_hash") != env["FOUNDATION_MODEL_HASH"] or
                        data.get("tokenizer_hash") != FOUNDATION["tokenizer.json"] or
                        data.get("adapter_hash", "").removeprefix("0x") != manifest["keccak256"].removeprefix("0x")):
                    raise ValueError("runtime identity mismatch")
                if len(data.get("results", [])) != len(batch):
                    raise ValueError("runtime returned wrong result count")
                bits = [r["score"] for r in data["results"]]
                for value in bits:
                    decode_bits(value)
                return bits, data["adapter_hash"]
            vectors = []
            for row in rows:
                for side in ("chosen", "rejected"):
                    text = f"### Task:\n{row['prompt']}\n\n### Answer:\n{row[side]}"
                    vectors.append(tokenizer.encode(text).ids)
            results, identity = [], None
            for start in range(0, len(vectors), 8):
                bits, adapter_hash = score(vectors[start:start + 8])
                if identity is not None and identity != adapter_hash:
                    raise ValueError("runtime adapter identity changed")
                identity = adapter_hash
                results.extend(bits)
            repeated, _ = score(vectors[:8])
            if repeated != results[:8]:
                raise ValueError("runtime score bits changed on identical inputs")
            wins = [decode_bits(results[2*i]) > decode_bits(results[2*i+1]) for i in range(len(rows))]
            by_label = {label: sum(w for r, w in zip(rows, wins) if r["label"] == label) /
                        sum(r["label"] == label for r in rows) for label in {r["label"] for r in rows}}
            accuracy = sum(wins) / len(wins)
            passed = accuracy >= 0.80 and min(by_label.values()) >= 0.70
            evidence = {"family": family, "verdict": "PASS" if passed else "FAIL", "originals": len(rows),
                        "runtime_pairwise_accuracy": accuracy, "accuracy_by_label": by_label,
                        "repeat_bit_vectors": 8, "adapter_hash": identity, "foundation_hash": env["FOUNDATION_MODEL_HASH"],
                        "tokenizer_hash": FOUNDATION["tokenizer.json"], "node_qualification": "not attempted"}
            write_json(root / "runtime-result.json", evidence)
            print(json.dumps(evidence))
            if not passed:
                raise ValueError("exported reference-runtime held-out gate failed")
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)


if __name__ == "__main__":
    main()
