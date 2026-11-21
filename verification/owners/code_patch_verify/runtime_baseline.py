# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Bounded public CPU untrained execution control. Never a registration candidate."""
import argparse
import hashlib
import json
import math
import os
import socket
import struct
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

from remote import remote_only

PIN = "1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd"
KIT_SHA = "18d7dbe7425e977b576a70146f901a006b8e4abddbfd351276205faa7f287243"
RELEASE = "https://github.com/telegraphprotocol/telegraph-foundation/releases/download/modernbert-base-lora16-v1/"
ASSETS = {
    "foundation_lora_fp32.onnx": ("0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6", 599117769),
    "foundation_engine.wasm": ("77643d6e1176ab8744801cef93ccb844d0136ed32b8daa865d541f37d13292de", 2686528),
    "foundation.json": ("f7de64b999acd3f735118fbe5999abf251fdf188f9797e3a4e36c54ee9e4dbbe", 1369),
    "tokenizer.json": ("67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456", 3583486),
}
TARGETS = ("attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo")
HERE = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_spec(spec):
    if (not isinstance(spec, dict) or spec.get("foundation_hash") != ASSETS["foundation_lora_fp32.onnx"][0]
            or spec.get("tokenizer_hash") != ASSETS["tokenizer.json"][0]
            or spec.get("foundation_version") != "modernbert-base-lora16-v1"
            or any(type(spec.get(key)) is not int or not 1 <= spec[key] <= maximum for key, maximum in
                   (("r_max", 16), ("num_layers", 64), ("hidden_size", 4096)))
            or not isinstance(spec.get("targets"), dict) or set(spec["targets"]) != set(TARGETS)):
        raise ValueError("pinned bounded released foundation layout required")
    for dims in spec["targets"].values():
        if not isinstance(dims, list) or len(dims) != 2 or any(type(n) is not int or not 1 <= n <= 16384 for n in dims):
            raise ValueError("native bounded target dimensions required")


def input_root(batch):
    rows = []
    for index, tokens in enumerate(batch):
        if (not isinstance(tokens, list) or not 1 <= len(tokens) <= 512
                or any(type(token) is not int or not 0 <= token < 50368 for token in tokens)):
            raise ValueError("bounded literal token vector required")
        raw = json.dumps({"input_ids": tokens, "attention_mask": [1] * len(tokens)}, separators=(",", ":")).encode()
        rows.append(str(index) + ":" + hashlib.sha256(raw).hexdigest())
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def response_bits(value, identity, root, count):
    if (not isinstance(value, dict) or set(value) != {"results", "foundation_hash", "adapter_hash", "tokenizer_hash", "input_root"}
            or any(value.get(key) != expected for key, expected in identity.items()) or value.get("input_root") != root
            or not isinstance(value.get("results"), list) or len(value["results"]) != count):
        raise ValueError("literal executor identity/root/positional count mismatch")
    bits = []
    for item in value["results"]:
        if (not isinstance(item, dict) or set(item) != {"score"} or type(item["score"]) is not int
                or not 0 <= item["score"] < 1 << 32 or item["score"] & 0x7f800000 == 0x7f800000):
            raise ValueError("actual finite UInt32 score bits required")
        bits.append(item["score"])
    return bits


def fetch_hash(url, path, expected, limit, exact_size=None):
    remote_only()
    digest, total, deadline = hashlib.sha256(), 0, time.monotonic() + 900
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "cpv-public-cpu-baseline/1.0"}), timeout=60) as response, path.open("wb") as stream:
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("asset download wall-time ceiling")
            chunk = response.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise ValueError("asset byte ceiling")
            digest.update(chunk)
            stream.write(chunk)
    if digest.hexdigest() != expected or exact_size is not None and total != exact_size:
        raise ValueError("released asset hash/size differs")


def write_control_package(path, spec):
    remote_only()
    check_spec(spec)
    terms = json.loads((HERE / "runtime_assets/licences.json").read_text())
    meta = {"format_version": "1", "abi": "lora-v1", "class": "judgment", "intent": "CODE_PATCH_VERIFY",
            "foundation_hash": spec["foundation_hash"], "foundation_version": spec["foundation_version"],
            "tokenizer_hash": spec["tokenizer_hash"], "rank": "1", "alpha": "1", "max_tokens": "512",
            "target_modules": '["Wi","Wo","Wqkv"]',
            "x-cpv-role": "UNTRAINED_EXECUTION_CONTROL_NEVER_REGISTER",
            "x-own-sand": terms["sand"], "x-upstream-apache-2.0": terms["apache_2_0"],
            "x-upstream-mit": terms["kit_mit"], "x-upstream-notice": (HERE / "NOTICE").read_text()}
    entries, size = {"__metadata__": meta}, 0
    shapes = []
    for layer in range(spec["num_layers"]):
        for target in TARGETS:
            incoming, outgoing = spec["targets"][target]
            shapes.extend(((f"lora.{layer}.{target}.A", [incoming, 1]), (f"lora.{layer}.{target}.B", [1, outgoing])))
    shapes.extend((("head.weight", [spec["hidden_size"], 1]), ("head.bias", [1])))
    for name, shape in shapes:
        length = math.prod(shape) * 4
        entries[name] = {"dtype": "F32", "shape": shape, "data_offsets": [size, size + length]}
        size += length
    header = json.dumps(entries, separators=(",", ":"), allow_nan=False).encode()
    header += b" " * (-len(header) % 8)
    if len(header) > 1 << 20 or size + len(header) + 8 > 64 << 20:
        raise ValueError("control package ceiling")
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(size))


def execute(root, out):
    remote_only()
    runner = Path(os.environ["RUNNER_TEMP"]).resolve()
    if not root.resolve().is_relative_to(runner) or not out.resolve().is_relative_to(runner):
        raise ValueError("remote private state must stay in RUNNER_TEMP")
    root.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    for name, (digest, count) in ASSETS.items():
        fetch_hash(RELEASE + name, root / name, digest, count, count)
    spec = json.loads((root / "foundation.json").read_text())
    check_spec(spec)
    archive = root / "kit.tar.gz"
    fetch_hash("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + PIN,
               archive, KIT_SHA, 32 << 20)
    with tarfile.open(archive) as source:
        members = source.getmembers()
        if len(members) > 500 or sum(member.size for member in members) > 128 << 20:
            raise ValueError("pinned source archive ceiling")
        source.extractall(root, filter="data")
    kit = root / ("telegraph-wasm-baseline-" + PIN)
    package = root / "CODE_PATCH_VERIFY-UNTRAINED-CONTROL.safetensors"
    write_control_package(package, spec)
    subprocess.run(["go", "build", "-mod=readonly", "-o", str(root / "adapterexec"), "./cmd/adapterexec"], cwd=kit, check=True, timeout=600)
    parser_result = root / "parser-result.json"
    subprocess.run(["go", "run", "-mod=readonly", str(HERE / "baseline_check.go"), "--package", str(package),
                    "--sha256", sha(package), "--foundation", str(root / "foundation.json"), "--out", str(parser_result)],
                   cwd=kit, check=True, timeout=600)
    parsed = json.loads(parser_result.read_text())
    if parsed.get("full_parser_passed") is not True or parsed.get("package_sha256") != sha(package):
        raise ValueError("actual unmodified full parser result required")
    identity = {"foundation_hash": spec["foundation_hash"], "adapter_hash": parsed["package_keccak"],
                "tokenizer_hash": spec["tokenizer_hash"]}
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    environment = {"PATH": os.environ["PATH"], "HOME": str(root), "FOUNDATION_POOL_SIZE": "1",
                   "FOUNDATION_MODEL_PATH": str(root / "foundation_lora_fp32.onnx"),
                   "FOUNDATION_MODEL_HASH": spec["foundation_hash"],
                   "FOUNDATION_ENGINE_WASM_PATH": str(root / "foundation_engine.wasm"),
                   "FOUNDATION_ENGINE_WASM_HASH": ASSETS["foundation_engine.wasm"][0]}
    # This job scores only fixed original source vectors, never evaluation inputs.
    startup_log = (root / "executor-fixed-control.log").open("wb")
    process = subprocess.Popen([str(root / "adapterexec"), "--package", str(package), "--listen", f"127.0.0.1:{port}"],
                               cwd=root, env=environment, stdout=subprocess.DEVNULL, stderr=startup_log)
    try:
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if process.poll() is not None:
                startup_log.flush()
                diagnostic = (root / "executor-fixed-control.log").read_bytes()
                if len(diagnostic) > 65536:
                    raise RuntimeError("reference executor startup diagnostic exceeded its ceiling")
                raise RuntimeError("reference executor exited during foundation load: " + diagnostic.decode("utf-8", errors="replace"))
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError("reference foundation startup ceiling")

        def post(value):
            raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
            request = urllib.request.Request(f"http://127.0.0.1:{port}/runnable", data=raw, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=600) as response:
                raw = response.read((1 << 20) + 1)
                if response.status != 200 or len(raw) > 1 << 20:
                    raise ValueError("reference response ceiling/status")
            return json.loads(raw)

        tokens = [[0], [1], [2], [50281, 0, 50282], [50281, 1, 2, 50282], [1024, 2048], [50], [3, 2, 1]]
        root_hash = input_root(tokens)
        request = {**identity, "input_root": root_hash, "batch": [{"tokens": vector} for vector in tokens]}
        first = response_bits(post(request), identity, root_hash, len(tokens))
        if response_bits(post(request), identity, root_hash, len(tokens)) != first:
            raise ValueError("real repeated reference score bits differ")
        for key in identity:
            bad = {**identity, key: "0" * 64, "batch": [{"tokens": [1]}]}
            reply = post(bad)
            if any(reply.get(name) != value for name, value in identity.items()) or reply[key] == bad[key]:
                raise ValueError("counterparty literal mismatch control did not echo actual identity")
        for vector in ([], [-1], [50368]):
            reply = post({**identity, "batch": [{"tokens": vector}]})
            if len(reply.get("results", [])) != 1 or not isinstance(reply["results"][0].get("error"), str):
                raise ValueError("actual reference invalid-token control accepted")
        save(out / "runtime-baseline.json", {"schema_version": 1, "status": "REFERENCE_ENGINE_BASELINE_ONLY", "family": "CODE_PATCH_VERIFY",
             "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
             "package_sha256": sha(package), "package_keccak": parsed["package_keccak"],
             "foundation_sha256": spec["foundation_hash"], "engine_sha256": ASSETS["foundation_engine.wasm"][0],
             "tokenizer_sha256": spec["tokenizer_hash"], "foundation_spec_sha256": ASSETS["foundation.json"][0],
             "full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
             "literal_negative_identity_echo_controls": 3, "actual_invalid_token_controls": 3,
             "vectors": [{"tokens": vector, "score_bits": bits} for vector, bits in zip(tokens, first)],
             "input_root": root_hash, "runtime": "pinned upstream adapterexec", "runner_os": os.environ["RUNNER_OS"],
             "runner_arch": os.environ["RUNNER_ARCH"], "registration_ready": False, "training_performed": False,
             "limits": ["Untrained zero-LoRA/zero-head execution control. Never register this package or claim quality from these source vectors.",
                        "Wrong-identity controls measure actual bare-hash echoes, not HTTP refusals: that is the literal upstream contract.",
                        "No training, independent factual audit, protected pool access, serving lease or signing occurred."]})
        save(out / "parser-result.json", parsed)
        shutil_package = out / package.name
        shutil_package.write_bytes(package.read_bytes())
        save(out / "licences.json", json.loads((HERE / "runtime_assets/licences.json").read_text()))
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        startup_log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.check_only:
        try:
            remote_only()
        except RuntimeError:
            print("PASS: baseline model creation/loading/inference refused before any local work")
        return
    remote_only()
    if args.root is None or args.out is None:
        raise ValueError("remote root and output paths required")
    execute(args.root, args.out)


if __name__ == "__main__":
    main()
