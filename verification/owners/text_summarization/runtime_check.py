# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Remote-only full parser, export ordering and fresh-process runtime gates."""
import json
import math
import os
import socket
import subprocess
import time
import urllib.request
from contextlib import contextmanager

from data import HERE, VIEWS, digest, read_splits, render, view, write_json
from live_class import CHILD, FAMILY, capture
from metrics import pair_metrics
from wire import decode_bits, input_root, validate_response

RELEASE_HASHES = {
    "foundation_lora_fp32.onnx": "0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6",
    "foundation_engine.wasm": "77643d6e1176ab8744801cef93ccb844d0136ed32b8daa865d541f37d13292de",
    "tokenizer.json": "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456",
}
MAX_RUNTIME_SECONDS = 7200
BATCH = 4
MAX_LOG_BYTES = 16 << 20
ATOL, RTOL = 1e-3, 1e-3


def model_remote_only():
    if (os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_REPOSITORY") != "zkasuran/telegraph-catalog-miners"
            or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"):
        raise RuntimeError("runtime scoring is limited to the manually dispatched public CPU job")


def metrics(rows, bits):
    if len(rows) != 128 or len({r["group_id"] for r in rows}) != 64 or len(bits) != 256:
        raise ValueError("runtime audit needs all 64 independent reciprocal groups")
    return pair_metrics(rows, [decode_bits(b) for b in bits])


def native_agreement(native, bits):
    if len(native) != len(bits) or len(native) != 256:
        raise ValueError("native/exported audit score counts differ")
    values = [decode_bits(b) for b in bits]
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in native):
        raise ValueError("nonfinite native score")
    errors = [abs(x - y) for x, y in zip(native, values)]
    numerical = all(error <= ATOL + RTOL * abs(x) for x, error in zip(native, errors))
    ordering = all((native[i] > native[i + 1]) == (values[i] > values[i + 1]) for i in range(0, len(native), 2))
    return {"numerical": numerical, "ordering": ordering, "atol": ATOL, "rtol": RTOL,
            "max_absolute_error": max(errors), "exact_bits_claimed": False}


def compile_tools(root, kit, go="go"):
    environment = {**os.environ, "GOMAXPROCS": "2"}
    subprocess.run([go, "test", "-p=1", "-mod=readonly", str(HERE / "packagecheck.go"), str(HERE / "packagecheck_test.go")],
                   cwd=kit, env=environment, check=True, timeout=600)
    for name, source in (("packagecheck", str(HERE / "packagecheck.go")), ("adapterexec", "./cmd/adapterexec")):
        subprocess.run([go, "build", "-p=1", "-mod=readonly", "-trimpath", "-o", str(root / name), source],
                        cwd=kit, env=environment, check=True, timeout=600)


def preflight(root):
    captured = capture(root)
    command = [str(root / "packagecheck"), "--intent", FAMILY, "--child", CHILD,
               "--class-snapshot", str(root / "class-snapshot.json"),
               "--snapshot-sha256", captured["snapshot_sha256"], "--preflight"]
    completed = subprocess.run(command, capture_output=True, timeout=60, check=True)
    value = json.loads(completed.stdout)
    if value.get("class") != "judgment" or value.get("loaded") is not False:
        raise ValueError("live-class source preflight mismatch")
    write_json(root / "class-preflight.json", value)
    return captured


def validate_package(root):
    captured = capture(root)
    path = root / (FAMILY + ".safetensors")
    expected = digest(path.read_bytes())
    command = [str(root / "packagecheck"), "--intent", FAMILY, "--child", CHILD,
               "--class-snapshot", str(root / "class-snapshot.json"),
               "--snapshot-sha256", captured["snapshot_sha256"], "--package", str(path), "--sha256", expected,
               "--foundation", str(root / "foundation.json"),
               "--foundation-sha256", digest((root / "foundation.json").read_bytes()), "--out", str(root)]
    completed = subprocess.run(command, capture_output=True, timeout=60, check=False)
    (root / "package-check.stderr.txt").write_bytes(completed.stderr[:1 << 20])
    (root / "package-check.stdout.txt").write_bytes(completed.stdout[:1 << 20])
    if completed.returncode or max(len(completed.stdout), len(completed.stderr)) > 1 << 20:
        raise ValueError("full reference Stage 1 parser refused package")
    manifest = json.loads(completed.stdout)
    if manifest["sha256"] != expected or manifest["manifest"]["verification"] != {"stage1": True, "loaded": False}:
        raise ValueError("full Stage 1 package binding mismatch")
    write_json(root / "package-manifest.json", manifest)
    return manifest


@contextmanager
def server(root, manifest, suffix, overrides=None):
    model_remote_only()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    foundation = root / "foundation"
    env = {**os.environ, "FOUNDATION_MODEL_PATH": str(foundation / "foundation_lora_fp32.onnx"),
           "FOUNDATION_MODEL_HASH": RELEASE_HASHES["foundation_lora_fp32.onnx"],
           "FOUNDATION_ENGINE_WASM_PATH": str(foundation / "foundation_engine.wasm"),
           "FOUNDATION_ENGINE_WASM_HASH": RELEASE_HASHES["foundation_engine.wasm"],
           "ADAPTER_SCORE_TIMEOUT": "180s"}
    env.update(overrides or {})
    logpath = root / ("runtime-" + suffix + ".log")
    with logpath.open("w") as log:
        child = subprocess.Popen([str(root / "adapterexec"), "--package", str(root / (FAMILY + ".safetensors")),
                                  "--listen", f"127.0.0.1:{port}"], env=env, stdout=log, stderr=log)
        try:
            limit = time.monotonic() + 180
            while time.monotonic() < limit:
                if child.poll() is not None:
                    raise RuntimeError("reference executor failed to load")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    time.sleep(.5)
            else:
                raise TimeoutError("reference executor readiness ceiling")
            deadline = time.monotonic() + MAX_RUNTIME_SECONDS

            def score(batch, identity=None, refusal=False):
                if not 1 <= len(batch) <= BATCH:
                    raise ValueError("fixed runtime batch ceiling")
                if not refusal and any(not isinstance(ids, list) or not 1 <= len(ids) <= 256 or
                       any(type(token) is not int or not 0 <= token < 50368 for token in ids) for ids in batch):
                    raise ValueError("invalid canonical token vector")
                remaining = deadline - time.monotonic()
                if remaining <= 0 or logpath.stat().st_size > MAX_LOG_BYTES:
                    raise TimeoutError("reference runtime time or diagnostic ceiling")
                payload = {"batch": [{"tokens": ids} for ids in batch],
                           "foundation_hash": RELEASE_HASHES["foundation_lora_fp32.onnx"],
                           "tokenizer_hash": RELEASE_HASHES["tokenizer.json"],
                           "adapter_hash": manifest["keccak256"].removeprefix("0x")}
                payload.update(identity or {})
                payload["input_root"] = input_root(payload["batch"])
                req = urllib.request.Request(f"http://127.0.0.1:{port}/runnable", data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=min(180, remaining)) as response:
                    raw = response.read((1 << 20) + 1)
                    if response.status != 200 or len(raw) > 1 << 20:
                        raise ValueError("runtime response status or ceiling")
                data = json.loads(raw)
                return validate_response(data, payload, refusals=refusal)

            yield score
        finally:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)


def execute_runtime(root, manifest, native):
    model_remote_only()
    from tokenizers import Tokenizer
    _, splits = read_splits(root / "data")
    rows = {style: [view(row, style) for row in splits["audit"]] for style in VIEWS}
    for name, expected in RELEASE_HASHES.items():
        h = __import__("hashlib").sha256()
        with (root / "foundation" / name).open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != expected:
            raise ValueError("runtime foundation identity mismatch: " + name)
    tokenizer = Tokenizer.from_file(str(root / "foundation/tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    vectors = {style: [tokenizer.encode(render(row["prompt"], row[side])).ids for row in formatted for side in ("chosen", "rejected")]
               for style, formatted in rows.items()}
    if any(len(v) != 256 or any(not ids or len(ids) > 256 for ids in v) for v in vectors.values()):
        raise ValueError("runtime input is missing or clipped")
    started = time.monotonic()
    streams = {}
    fixed = [[50281] + [22 + ((i + n) % 100) for i in range(n - 2)] + [50282] for n in (4, 8, 16, 32, 64, 128, 192, 256)]
    with server(root, manifest, "first") as score:
        for style, batch in vectors.items():
            streams[style] = [bit for start in range(0, len(batch), BATCH) for bit in score(batch[start:start + BATCH])]
            write_json(root / "runtime-score-bits.json", {"formats": streams, "token_vectors_sha256": digest(json.dumps(vectors).encode())})
        first = [bit for start in range(0, 8, BATCH) for bit in score(fixed[start:start + BATCH])]
        repeated = [bit for start in range(0, 8, BATCH) for bit in score(fixed[start:start + BATCH])]
        failures = 0
        for key in ("foundation_hash", "tokenizer_hash", "adapter_hash"):
            try:
                score(fixed[:1], {key: "0" * 64})
            except ValueError:
                failures += 1
            else:
                raise ValueError("wrong runtime identity escaped verification")
        for invalid in ([], [-1]):
            score([invalid], refusal=True)
    if first != repeated:
        raise ValueError("repeated exact input bits differ")
    with server(root, manifest, "fresh") as score:
        fresh = [bit for start in range(0, 8, BATCH) for bit in score(fixed[start:start + BATCH])]
    if fresh != first:
        raise ValueError("fresh-process engine score bits differ")
    agreement = {style: native_agreement(native[style], streams[style]) for style in VIEWS}
    result = {style: metrics(rows[style], streams[style]) for style in VIEWS}
    passed = all(agreement[style]["numerical"] and agreement[style]["ordering"] and
                 result[style]["pairwise_accuracy"] >= .8 and min(result[style]["subsets"].values()) >= .7 for style in VIEWS)
    report = {"verdict": "PASS" if passed else "FAIL", "metrics": result, "native_wasm": agreement,
               "audit_token_vectors": 768, "repeat_bit_vectors": 8, "fresh_process_bit_vectors": 8, "wrong_identity_rejections": failures,
               "vectors": [{"tokens": tokens, "score_bits": bits} for tokens, bits in zip(fixed, first)],
               "repeat_bits_identical": first == repeated == fresh,
               "invalid_token_controls": 2, "literal_bare_hash_echoes": True,
              "foundation": RELEASE_HASHES, "package_sha256": manifest["sha256"], "adapter_hash": manifest["keccak256"],
              "elapsed_seconds": time.monotonic() - started, "node_qualification": "not attempted"}
    write_json(root / "runtime-result.json", report)
    return report
