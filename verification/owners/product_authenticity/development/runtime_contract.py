# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Literal reference-executor wire checks. Neural execution is remote-only."""
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request


def remote_only():
    if (os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_REPOSITORY") != "zkasuran/telegraph-catalog-miners"
            or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch" or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"):
        raise RuntimeError("neural work requires the bounded owned standard public CPU workflow")


def score_value(bits):
    if type(bits) is not int or not 0 <= bits < 1 << 32:
        raise ValueError("invalid actual FP32 bits")
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    if not math.isfinite(value):
        raise ValueError("non-finite score bits")
    return value


def check_reply(reply, count, foundation, tokenizer, adapter):
    if not re.fullmatch(r"[a-f0-9]{64}", adapter):
        raise ValueError("expected adapter identity must be bare lowercase keccak")
    # Never normalise the counterparty's wire echo into a passing stand-in.
    for key, expected in (("foundation_hash", foundation), ("tokenizer_hash", tokenizer), ("adapter_hash", adapter)):
        if reply.get(key) != expected:
            raise ValueError("literal wire identity mismatch: " + key)
    results = reply.get("results")
    if not isinstance(results, list) or len(results) != count:
        raise ValueError("wire result count mismatch")
    bits = []
    for item in results:
        if not isinstance(item, dict) or set(item) != {"score"}:
            raise ValueError("executor item error or unknown result fields")
        score_value(item["score"])
        bits.append(item["score"])
    return bits


def execute(root, package, identity, vectors, max_tokens=256):
    remote_only()
    root = Path(root)
    if max_tokens not in (256, 512) or not 8 <= len(vectors) <= 100 or any(not v or len(v) > max_tokens for v in vectors):
        raise ValueError("8-100 bounded real fixed vectors required")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {**os.environ, "FOUNDATION_MODEL_PATH": str(root / "foundation/foundation_lora_fp32.onnx"),
           "FOUNDATION_MODEL_HASH": identity["foundation_hash"], "FOUNDATION_ENGINE_WASM_PATH": str(root / "foundation/foundation_engine.wasm"),
           "FOUNDATION_ENGINE_WASM_HASH": identity["engine_hash"], "ADAPTER_SCORE_TIMEOUT": "180s"}
    log_path = root / "reference-runtime.log"
    def one_process(suffix):
        with log_path.open("a") as log:
            process = subprocess.Popen([str(root / "adapterexec"), "--package", str(package), "--listen", f"127.0.0.1:{port}"], env=env, stdout=log, stderr=log)
            try:
                limit = time.monotonic() + 180
                while time.monotonic() < limit:
                    if process.poll() is not None:
                        raise RuntimeError("reference engine load failed")
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=1):
                            break
                    except OSError:
                        time.sleep(.5)
                else:
                    raise TimeoutError("reference readiness timeout")
                def request(ids, overrides=None):
                    payload = {"batch": [{"tokens": item} for item in ids], "foundation_hash": identity["foundation_hash"],
                               "tokenizer_hash": identity["tokenizer_hash"], "adapter_hash": identity["adapter_hash"]}
                    payload.update(overrides or {})
                    req = urllib.request.Request(f"http://127.0.0.1:{port}/runnable", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=180) as response:
                        raw = response.read((1 << 20) + 1)
                    if len(raw) > 1 << 20:
                        raise ValueError("runtime response ceiling")
                    reply = json.loads(raw)
                    expected = {**identity, **(overrides or {})}
                    return check_reply(reply, len(ids), expected["foundation_hash"], expected["tokenizer_hash"], expected["adapter_hash"])
                first = [bit for i in range(0, len(vectors), 4) for bit in request(vectors[i:i + 4])]
                repeated = [bit for i in range(0, len(vectors), 4) for bit in request(vectors[i:i + 4])]
                if first != repeated:
                    raise ValueError("reference repeated bits differ")
                controls = 0
                if suffix == "first":
                    for key in ("foundation_hash", "tokenizer_hash", "adapter_hash"):
                        try:
                            request(vectors[:1], {key: "0" * 64})
                        except (ValueError, urllib.error.HTTPError):
                            controls += 1
                        else:
                            raise ValueError("negative identity control escaped literal verification")
                return first, controls
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    first, controls = one_process("first")
    fresh, _ = one_process("fresh")
    if first != fresh or controls != 3:
        raise ValueError("fresh engine or negative identity controls failed")
    return {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
            "negative_identity_controls": controls, "vectors": [{"tokens": ids, "score_bits": bits} for ids, bits in zip(vectors, first)],
            "audit_measured": False, "kind": "development fixed vectors, not a quality audit"}
