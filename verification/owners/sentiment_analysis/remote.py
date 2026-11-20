# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded public CPU audit of the existing frozen scorer and recovery bytes."""
import argparse
import os
import signal
import socket
import subprocess
import tarfile
import threading
import time
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from build import build
from contracts import (
    CHILD,
    FAMILY,
    FORMATS,
    FOUNDATION,
    FROZEN,
    HERE,
    MAX_BODY,
    NODE,
    OWNER,
    RELEASE,
    SPEC,
    fetch,
    file_sha,
    finite_bits,
    remote_only,
    strict_json,
    validate_ready,
    write_json,
)
from proxy import Backend, BoundedServer, identity, input_root, validate_response
from source_data import check_splits, pair, prepare, shortcuts


def classes(root):
    snapshot = {"schema": 1}
    for name, url in (("family", NODE + "/api/wasm?resolve=" + FAMILY), ("child", NODE + "/engine/v1/intents/" + CHILD)):
        request = urllib.request.Request(url, headers={"User-Agent": "Sentiment-owner-verification/1.0", "Cache-Control": "no-cache"})
        with urllib.request.urlopen(request, timeout=25) as response:
            raw = response.read(MAX_BODY + 1)
            if len(raw) > MAX_BODY or response.status != 200 or response.geturl() != url:
                raise ValueError("authoritative class source transport mismatch")
        snapshot[name] = {"url": url, "raw": raw.decode()}
    snapshot["fetched_at"] = datetime.now(timezone.utc).isoformat()
    from common import check_classes
    check_classes(snapshot, {"family": FAMILY, "child": CHILD, "class": "judgment"})
    write_json(root / "class-snapshot.json", snapshot)


def assets(root):
    remote_only()
    root.mkdir(parents=True, exist_ok=True)
    model = root / "foundation"
    model.mkdir(exist_ok=True)
    for name, expected in FOUNDATION.items():
        fetch(RELEASE + name, model / name, expected, 1 << 30)
    write_json(root / "foundation.json", SPEC)
    archive = root / "kit.tar.gz"
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + FROZEN["kit_commit"],
          archive, FROZEN["kit_archive_sha256"], 32 << 20)
    with tarfile.open(archive) as source:
        members = source.getmembers()
        if len(members) > 5000 or sum(item.size for item in members) > 128 << 20:
            raise ValueError("pinned kit archive ceiling")
        source.extractall(root, filter="data")
    kit = root / ("telegraph-wasm-baseline-" + FROZEN["kit_commit"])
    for name, source in (("stagecheck", str(HERE / "stagecheck.go")), ("adapterexec", "./cmd/adapterexec")):
        subprocess.run(["go", "build", "-mod=readonly", "-trimpath", "-o", str(root / name), source],
                       cwd=kit, check=True, timeout=600)
    return model


def stage_check(root, package, expected_sha, label):
    classes(root)
    command = [str(root / "stagecheck"), "--intent", FAMILY, "--child", CHILD, "--expected-class", "judgment",
               "--package", str(package), "--sha256", expected_sha, "--foundation", str(root / "foundation.json"),
               "--foundation-sha256", file_sha(root / "foundation.json"), "--class-snapshot", str(root / "class-snapshot.json"),
               "--snapshot-sha256", file_sha(root / "class-snapshot.json"), "--out", str(root)]
    with (root / (label + ".stdout")).open("wb") as out, (root / (label + ".stderr")).open("wb") as err:
        result = subprocess.run(command, stdout=out, stderr=err, check=False, timeout=60)
    if result.returncode or (root / (label + ".stdout")).stat().st_size > MAX_BODY:
        raise ValueError("full pinned package parser refused")
    manifest = strict_json((root / (label + ".stdout")).read_bytes())
    if manifest["sha256"] != expected_sha or manifest["manifest"]["verification"] != {"stage1": True, "loaded": False}:
        raise ValueError("full parser/static identity mismatch")
    write_json(root / (label + ".json"), manifest)
    return manifest


def fixed_vectors():
    return [[50281] + [22 + (i % 100) for i in range(length - 2)] + [50282]
            for length in (4, 8, 16, 32, 64, 96, 128, 256)]


@contextmanager
def executor(root, package, keccak):
    remote_only()
    for name, expected in FOUNDATION.items():
        if file_sha(root / "foundation" / name) != expected:
            raise ValueError("actual runtime asset hash mismatch")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    environment = {"PATH": os.environ["PATH"], "HOME": str(root), "FOUNDATION_POOL_SIZE": "1",
                   "FOUNDATION_MODEL_PATH": str(root / "foundation/foundation_lora_fp32.onnx"),
                   "FOUNDATION_MODEL_HASH": FOUNDATION["foundation_lora_fp32.onnx"],
                   "FOUNDATION_ENGINE_WASM_PATH": str(root / "foundation/foundation_engine.wasm"),
                   "FOUNDATION_ENGINE_WASM_HASH": FOUNDATION["foundation_engine.wasm"]}
    process = subprocess.Popen([str(root / "adapterexec"), "--package", str(package), "--listen", f"127.0.0.1:{port}"],
                               env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    server = None
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("actual reference executor exited during load")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("reference engine load timeout")
        row = {"keccak": keccak, "runnable_url": "https://tga1.zkasuran.dev/_s2owner/runnable"}
        backend = Backend(port)
        server = BoundedServer(("127.0.0.1", 0), row, backend)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield backend, row, server
    finally:
        if server:
            server.shutdown()
            server.server_close()
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def score(backend, row, vectors):
    results = []
    for start in range(0, len(vectors), 8):
        batch = [{"tokens": tokens} for tokens in vectors[start:start + 8]]
        request = {"batch": batch, **identity(row), "input_root": input_root(batch)}
        response = validate_response(backend.score(request), request, row)
        # This is the public client's literal contract, not a normalized stand-in.
        if response["adapter_hash"] != row["keccak"]:
            raise ValueError("literal bare-hash echo mismatch")
        results.extend(item["score"] for item in response["results"])
    return results


def identity_controls(server, row):
    import http.client

    from common import json_bytes
    batch = [{"tokens": [50281, 22, 23, 50282]}]
    count = 0
    for field in ("foundation_hash", "adapter_hash", "tokenizer_hash"):
        body = {"batch": batch, **identity(row), field: "0"*64}
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        try:
            connection.request("POST", "/runnable", json_bytes(body), {"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 400 or strict_json(response.read(MAX_BODY)) != {"error": "request refused"}:
                raise ValueError("negative identity control failed")
            count += 1
        finally:
            connection.close()
    for tokens in ([], [-1]):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        try:
            connection.request("POST", "/runnable", json_bytes({"batch": [{"tokens": tokens}], **identity(row)}),
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 400:
                raise ValueError("invalid token refusal failed")
            response.read(MAX_BODY)
        finally:
            connection.close()
    return count


def audit(root):
    remote_only()
    model = assets(root)
    preservation = build(root)
    if preservation["package_sha256"] != FROZEN["recovered_sha256"]:
        raise ValueError("frozen recovery bytes changed")
    originals, source = prepare(root / "data")
    check_splits(originals)
    coefficient = strict_json((HERE / "frozen-baseline.json").read_bytes())
    if file_sha(HERE / "frozen-baseline.json") != FROZEN["baseline_sha256"] or coefficient["train_sha256"] != FROZEN["splits"]["train"]["sha256"]:
        raise ValueError("immutable train-fitted baseline changed")
    if file_sha(HERE / "licences.json") != FROZEN["licences_sha256"]:
        raise ValueError("retained grant binding changed")
    frozen_proof = root / "prior-runtime.json"
    fetch(FROZEN["runtime_url"], frozen_proof, FROZEN["runtime_sha256"], MAX_BODY)
    prior = strict_json(frozen_proof.read_bytes())
    if prior["verdict"] != "PASS" or prior["package_sha256"] != FROZEN["origin_sha256"] or prior["recovery_identity"] != {
            "run_id": str(FROZEN["runtime_run_id"]), "commit": FROZEN["runtime_commit"]}:
        raise ValueError("existing real runtime proof identity mismatch")
    original_manifest = stage_check(root, root / "original.safetensors", FROZEN["origin_sha256"], "original-parser")
    recovered = root / "SENTIMENT_ANALYSIS-recovery.safetensors"
    recovered_manifest = stage_check(root, recovered, preservation["package_sha256"], "recovered-parser")
    views, token_views = {}, {}
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    for style, name in enumerate(FORMATS):
        views[name] = [pair({**row, "text": strict_json(row["prompt"])["text"]}, style) for row in originals["test"]]
        token_views[name] = []
        for row in views[name]:
            for side in ("chosen", "rejected"):
                ids = tokenizer.encode(f"### Task:\n{row['prompt']}\n\n### Answer:\n{row[side]}").ids
                if not 1 <= len(ids) <= 256:
                    raise ValueError("audit differing claim would exceed guest token limit")
                token_views[name].append(ids)
    old_bits = {}
    # Sequential engines avoid two simultaneous Foundation processes.
    with executor(root, root / "original.safetensors", original_manifest["keccak256"].removeprefix("0x")) as (backend, row, _):
        for name in FORMATS:
            old_bits[name] = score(backend, row, token_views[name])
    quality = {}
    new_keccak = recovered_manifest["keccak256"].removeprefix("0x")
    with executor(root, recovered, new_keccak) as (backend, row, server):
        negatives = identity_controls(server, row)
        actual_vectors = fixed_vectors()
        first = score(backend, row, actual_vectors)
        second = score(backend, row, actual_vectors)
        if first != second:
            raise ValueError("actual fixed-vector repeat bits changed")
        for name in FORMATS:
            bits = score(backend, row, token_views[name])
            if bits != old_bits[name]:
                raise ValueError("metadata recovery changed actual reference score bits")
            wins = [finite_bits(bits[2*i]) > finite_bits(bits[2*i+1]) for i in range(256)]
            by_label = {label: sum(win for original, win in zip(views[name], wins) if original["label"] == label) / 128
                        for label in ("Positive", "Negative")}
            baseline = shortcuts(views[name], coefficient)
            quality[name] = {"accuracy": sum(wins)/256, "per_label": by_label,
                             "untrained_accuracy": FROZEN["untrained_correct"][name]/256,
                             "untrained_scope": "Original frozen seed-matched PyTorch run, not a new untrained WASM measurement",
                             "shortcut_accuracy": max(baseline.values()), "shortcuts": baseline,
                             "length_range_utf8": [min(len(strict_json(r["prompt"])["text"].encode()) for r in views[name]),
                                                   max(len(strict_json(r["prompt"])["text"].encode()) for r in views[name])]}
        # The serving front end must reproduce the same literal score response.
        import http.client

        from common import json_bytes
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=180)
        try:
            request = {"batch": [{"tokens": ids} for ids in actual_vectors], **identity(row)}
            connection.request("POST", "/runnable", json_bytes(request), {"Content-Type": "application/json"})
            response = connection.getresponse()
            body = strict_json(response.read(MAX_BODY))
            if response.status != 200 or body["adapter_hash"] != new_keccak or [r["score"] for r in body["results"]] != first:
                raise ValueError("actual bounded serving wire contract failed")
        finally:
            connection.close()
    passed = all(m["accuracy"] >= .80 and min(m["per_label"].values()) >= .70 and
                 m["accuracy"] >= m["untrained_accuracy"] + .05 and m["accuracy"] >= m["shortcut_accuracy"] + .10
                 for m in quality.values())
    write_json(root / "actual-quality.json", {"passed": passed, "formats": quality})
    report = {"schema_version": 1, "status": "REGISTRATION_READY", "family": FAMILY, "owner": OWNER,
              "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"], "serving_commit": os.environ["GITHUB_SHA"],
              "package_sha256": preservation["package_sha256"], "package_keccak": new_keccak, "package_bytes": recovered.stat().st_size,
              "foundation_sha256": FOUNDATION["foundation_lora_fp32.onnx"], "engine_sha256": FOUNDATION["foundation_engine.wasm"],
              "tokenizer_sha256": FOUNDATION["tokenizer.json"], "recovery_id": FROZEN["recovery_id"],
              "original_training": {"run_id": FROZEN["training_run_id"], "commit": FROZEN["training_commit"],
                                    "fitted_material": True, "selection": "original validation split only; weights frozen before source audit exposure"},
              "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True,
                         "source_manifest_sha256": file_sha(root / "data/source-manifest.json"), "independent_groups": source["independent_audit_groups"],
                         "boundary": "Yelp domain excluded from original training/validation; frozen-score recovery replay, no tuning",
                         "baseline_train_sha256": coefficient["train_sha256"], "baseline_sha256": FROZEN["baseline_sha256"]},
              "quality": {"passed": passed, "audit_pairs": 256, "all_supported_formats_tested": True, "formats": quality},
              "runtime": {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
                          "negative_identity_controls": negatives, "original_recovery_score_bits_identical": True,
                          "tensor_payload_sha256": preservation["tensor_payload_sha256"],
                          "vectors": [{"tokens": ids, "score_bits": bit} for ids, bit in zip(actual_vectors, first)]},
              "limits": [FROZEN["scope"], "Source annotations and constructed preferences are distinct; no human review claimed."]}
    validate_ready(report)
    write_json(root / "registration-ready.json", report)
    print("REGISTRATION_READY " + new_keccak)


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    root = args.root.resolve()
    try:
        audit(root)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.SubprocessError, tarfile.TarError) as error:
        if root.is_dir():
            write_json(root / "failure.json", {"status": "FAILED", "family": FAMILY, "error_type": type(error).__name__,
                       "run_id": os.environ.get("GITHUB_RUN_ID"), "commit": os.environ.get("GITHUB_SHA")})
        raise


if __name__ == "__main__":
    main()
