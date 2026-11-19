# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Actual reference executor checks, limited to the remote CPU workflow."""
import datetime
import json
import math
import os
import socket
import struct
import subprocess
import time
import urllib.request
from contextlib import contextmanager

from data import HERE, audit_labels, digest, write_json
from scorer import render

HASHES = {"foundation_lora_fp32.onnx": "0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6",
          "tokenizer.json": "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456",
          "foundation_engine.wasm": "77643d6e1176ab8744801cef93ccb844d0136ed32b8daa865d541f37d13292de"}
NODE = "https://devnode.telegraphprotocol.com"


class NegativeIdentityMismatch(ValueError):
    """The actual finite reply is bound, but it did not echo the requested identity."""


def compile_tools(root, kit):
    for files in (("wire_test.go",), ("packagecheck.go", "packagecheck_test.go")):
        subprocess.run(["go", "test", "-mod=readonly", *[str(HERE / name) for name in files]],
                       cwd=kit, check=True, timeout=600)
    for name, source in (("packagecheck", str(HERE / "packagecheck.go")), ("adapterexec", "./cmd/adapterexec")):
        subprocess.run(["go", "build", "-mod=readonly", "-trimpath", "-o", str(root / name), source],
                       cwd=kit, check=True, timeout=600)


def get(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(url, headers={"User-Agent": "email-runtime-proof/1"}), timeout=25) as response:
        raw = response.read((1 << 20) + 1)
        if response.status != 200 or len(raw) > 1 << 20:
            raise ValueError("source response status or byte ceiling")
        return raw.decode("utf-8")


def capture_class(root):
    snapshot = {"schema": 1, "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "family": {"url": NODE + "/api/wasm?resolve=EMAIL_SECURITY"},
                "child": {"url": NODE + "/engine/v1/intents/EMAIL_SECURITY_GENERAL"}}
    for key in ("family", "child"):
        snapshot[key]["raw"] = get(snapshot[key]["url"])
    child = json.loads(snapshot["child"]["raw"])
    if (child["class"] != "judgment" or child["request"]["schema_version"] != 3
            or [p["name"] for p in child["request"]["params"]] != ["domain", "record_type"]
            or child["response"]["fields"][0]["name"] != "lookup_ok"):
        raise ValueError("exact live EMAIL_SECURITY task changed")
    write_json(root / "class-snapshot.json", snapshot)
    sha = digest((root / "class-snapshot.json").read_bytes())
    value = subprocess.run([str(root / "packagecheck"), "--intent", "EMAIL_SECURITY", "--child", "EMAIL_SECURITY_GENERAL",
                            "--class-snapshot", str(root / "class-snapshot.json"), "--snapshot-sha256", sha, "--preflight"],
                           capture_output=True, text=True, check=True, timeout=60)
    result = json.loads(value.stdout)
    if result.get("class") != "judgment" or result.get("loaded") is not False:
        raise ValueError("class preflight binding failed")
    write_json(root / "class-preflight.json", result)
    return {"snapshot_sha256": sha, "fetched_at": snapshot["fetched_at"]}


def validate_package(root):
    captured = capture_class(root)
    package = root / "EMAIL_SECURITY.safetensors"
    sha = digest(package.read_bytes())
    command = [str(root / "packagecheck"), "--intent", "EMAIL_SECURITY", "--child", "EMAIL_SECURITY_GENERAL",
               "--class-snapshot", str(root / "class-snapshot.json"), "--snapshot-sha256", captured["snapshot_sha256"],
               "--package", str(package), "--sha256", sha, "--foundation", str(root / "foundation.json"),
               "--foundation-sha256", digest((root / "foundation.json").read_bytes()), "--out", str(root)]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
    manifest = json.loads(result.stdout)
    if manifest["sha256"] != sha or manifest["manifest"]["verification"] != {"stage1": True, "loaded": False}:
        raise ValueError("full upstream package parser binding failed")
    write_json(root / "package-manifest.json", manifest)
    return manifest


def decode_bits(value):
    if type(value) is not int or not 0 <= value <= 0xffffffff:
        raise ValueError("invalid FP32 bits")
    number = struct.unpack("<f", struct.pack("<I", value))[0]
    if not math.isfinite(number):
        raise ValueError("nonfinite runtime score")
    return number


def check_response(body, count, manifest, expected_root=None):
    if not isinstance(body, dict):
        raise ValueError("runtime response must be an object")
    identities = {"foundation_hash": HASHES["foundation_lora_fp32.onnx"], "tokenizer_hash": HASHES["tokenizer.json"],
                  "adapter_hash": manifest["keccak256"].removeprefix("0x")}
    # The node's literal wire uses bare lowercase hashes. A normalized stand-in
    # would miss the same 0x-prefixed echo defect that broke prior endpoints.
    if any(body.get(k) != v for k, v in identities.items()):
        raise ValueError("runtime identity mismatch")
    if expected_root is not None and body.get("input_root") != expected_root:
        raise ValueError("runtime literal input root mismatch")
    results = body.get("results")
    if not isinstance(results, list) or len(results) != count:
        raise ValueError("runtime count mismatch")
    bits = []
    for row in results:
        if not isinstance(row, dict) or "error" in row or "score" not in row:
            raise ValueError("runtime item failure")
        decode_bits(row["score"])
        bits.append(row["score"])
    return bits


@contextmanager
def server(root, manifest, suffix, deadline):
    from remote import remote_only
    remote_only()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {**os.environ, "FOUNDATION_MODEL_PATH": str(root / "foundation/foundation_lora_fp32.onnx"),
           "FOUNDATION_MODEL_HASH": HASHES["foundation_lora_fp32.onnx"],
           "FOUNDATION_ENGINE_WASM_PATH": str(root / "foundation/foundation_engine.wasm"),
           "FOUNDATION_ENGINE_WASM_HASH": HASHES["foundation_engine.wasm"], "ADAPTER_SCORE_TIMEOUT": "180s"}
    with (root / ("runtime-" + suffix + ".log")).open("w") as log:
        process = subprocess.Popen([str(root / "adapterexec"), "--package", str(root / "EMAIL_SECURITY.safetensors"),
                                    "--listen", f"127.0.0.1:{port}"], env=env, stdout=log, stderr=log)
        try:
            limit = min(time.monotonic() + 180, deadline)
            while time.monotonic() < limit:
                if process.poll() is not None:
                    raise RuntimeError("reference executor startup failed")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    time.sleep(.5)
            else:
                raise TimeoutError("reference executor startup ceiling")

            def score(vectors, wrong=None):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not 1 <= len(vectors) <= 4:
                    raise TimeoutError("runtime time or batch ceiling")
                payload = {"batch": [{"tokens": vector} for vector in vectors],
                           "foundation_hash": HASHES["foundation_lora_fp32.onnx"], "tokenizer_hash": HASHES["tokenizer.json"],
                           "adapter_hash": manifest["keccak256"].removeprefix("0x")}
                from wire_proxy import input_root
                payload["input_root"] = input_root(payload["batch"])
                if wrong:
                    payload[wrong] = "0" * 64
                request = urllib.request.Request(f"http://127.0.0.1:{port}/runnable", data=json.dumps(payload).encode(),
                                                 headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=min(180, remaining)) as response:
                    raw = response.read((1 << 20) + 1)
                    if response.status != 200 or len(raw) > 1 << 20:
                        raise ValueError("runtime response ceiling")
                body = json.loads(raw)
                # The primary executor reports actual identity. It need not
                # reject the request, so this verifier checks the returned one.
                bits = check_response(body, len(vectors), manifest, payload["input_root"])
                if wrong and body.get(wrong) != payload[wrong]:
                    raise NegativeIdentityMismatch("requested identity not reproduced by actual bound finite reply")
                return bits

            yield score
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def execute_runtime(root, manifest, rows, native):
    from remote import remote_only
    remote_only()
    from tokenizers import Tokenizer
    if len(rows) != 128 or len(native) != 256:
        raise ValueError("full fixed audit required")
    for name, sha in HASHES.items():
        h = __import__("hashlib").sha256()
        with (root / "foundation" / name).open("rb") as source:
            for raw in iter(lambda: source.read(1 << 20), b""):
                h.update(raw)
        if h.hexdigest() != sha:
            raise ValueError("foundation artifact changed")
    tokenizer = Tokenizer.from_file(str(root / "foundation/tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    vectors = [tokenizer.encode(render(r["prompt"], r[side])).ids for r in rows for side in ("chosen", "rejected")]
    if any(not ids or len(ids) > 256 for ids in vectors):
        raise ValueError("runtime cannot clip inputs")
    deadline = time.monotonic() + 1800
    with server(root, manifest, "first", deadline) as score:
        bits = [b for i in range(0, len(vectors), 4) for b in score(vectors[i:i+4])]
        repeat = [b for i in range(0, 8, 4) for b in score(vectors[i:i+4])]
        failures = 0
        for identity in ("foundation_hash", "tokenizer_hash", "adapter_hash"):
            try:
                score(vectors[:1], identity)
            except NegativeIdentityMismatch:
                failures += 1
            else:
                raise ValueError("wrong identity escaped verifier")
    with server(root, manifest, "fresh", deadline) as score:
        fresh = [b for i in range(0, 8, 4) for b in score(vectors[i:i+4])]
    values = [decode_bits(b) for b in bits]
    numerical = all(abs(a-b) <= 1e-3 + 1e-3 * abs(a) for a, b in zip(native, values))
    ordering = all((native[i] > native[i+1]) == (values[i] > values[i+1]) for i in range(0, len(bits), 2))
    accuracy = sum(values[i] > values[i+1] for i in range(0, len(bits), 2)) / len(rows)
    wins = [values[2*i] > values[2*i+1] for i in range(len(rows))]
    labels = [audit_labels(row) for row in rows]
    per_label = {key: sum(win for ls, win in zip(labels, wins) if key in ls) / sum(key in ls for ls in labels)
                 for key in sorted({key for ls in labels for key in ls})}
    repeat_identical = repeat == fresh == bits[:8]
    result = {"passed": numerical and ordering and repeat_identical and accuracy >= .8 and min(per_label.values()) >= .7,
              "pairwise_accuracy": accuracy, "per_label": per_label, "score_bits": bits,
              "repeat_bits_identical": repeat_identical, "repeat_bit_vectors": 8, "fresh_process_bit_vectors": 8,
              "vectors": [{"tokens": tokens, "score_bits": bit} for tokens, bit in zip(vectors[:8], bits[:8])],
              "wrong_identity_controls": failures, "native_numerical_agreement": numerical,
              "native_ordering_agreement": ordering, "token_vectors_sha256": digest(json.dumps(vectors).encode()),
              "package_sha256": manifest["sha256"], "adapter_hash": manifest["keccak256"], "foundation": HASHES,
              "github_run_id": os.environ["GITHUB_RUN_ID"], "commit": os.environ["GITHUB_SHA"],
              "node_qualification": "not attempted"}
    write_json(root / "runtime-result.json", result)
    return result
