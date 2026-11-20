# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded remote reference-engine replay of the immutable failed-run package."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from contracts import (
    FAMILY,
    FOUNDATION,
    HERE,
    KIT,
    KIT_SHA,
    PACKAGE_KECCAK,
    PACKAGE_SHA,
    RELEASE,
    SOURCE_SHA,
    TRAIN_COMMIT,
    TRAIN_RUN,
    helper_scope,
    metrics,
    read_asset,
    remote_only,
    rows,
    sha,
    verify_assets,
)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fetch(url, path, expected, ceiling=1 << 30):
    if path.is_file() and sha(path) == expected:
        return
    started, total, digest = time.monotonic(), 0, hashlib.sha256()
    partial = path.with_suffix(".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "Licensed-translation-replay/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as target:
        if response.status != 200:
            raise ValueError("download status")
        while chunk := response.read(1 << 20):
            total += len(chunk)
            if total > ceiling or time.monotonic() - started > 600:
                raise ValueError("download byte/time ceiling")
            digest.update(chunk)
            target.write(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("download hash mismatch")
    partial.replace(path)


def prepare(root):
    remote_only()
    root.mkdir(parents=True, exist_ok=True)
    foundation = root / "foundation"
    foundation.mkdir(exist_ok=True)
    for name, expected in FOUNDATION.items():
        fetch(RELEASE + name, foundation / name, expected)
    archive = root / "kit.tar.gz"
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + KIT, archive, KIT_SHA, 32 << 20)
    with tarfile.open(archive) as source:
        if sum(m.size for m in source.getmembers()) > 128 << 20:
            raise ValueError("source kit expansion ceiling")
        source.extractall(root, filter="data")
    kit = root / ("telegraph-wasm-baseline-" + KIT)
    parser_source = read_asset("parser-template.go.txt").decode()
    old = 'const familyID = "SENTIMENT_ANALYSIS"'
    if parser_source.count(old) != 1:
        raise ValueError("controller parser specialization changed")
    generated = root / "packagecheck.go"
    generated.write_text(parser_source.replace(old, 'const familyID = "LANGUAGE_TRANSLATION"', 1))
    for name, target in (("packagecheck", str(generated)), ("adapterexec", "./cmd/adapterexec")):
        subprocess.run(["go", "build", "-mod=readonly", "-trimpath", "-o", str(root / name), target], cwd=kit, check=True, timeout=600)
    package = root / (FAMILY + ".safetensors")
    shutil.copyfile(HERE / "assets/candidate.safetensors", package)
    if sha(package) != PACKAGE_SHA:
        raise ValueError("package copy changed bytes")
    config = foundation / "config.json"
    fetch("https://huggingface.co/answerdotai/ModernBERT-base/resolve/8949b909ec900327062f0ebf497f51aef5e6f0c8/config.json", config,
          "1609d59e627c33eaed524b4f01e546d42e84190a079a5a5ded84b212c41c324f", 32 << 10)
    cfg = json.loads(config.read_text())
    h, intermediate = cfg["hidden_size"], cfg["intermediate_size"]
    layout = {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"], "tokenizer_hash": FOUNDATION["tokenizer.json"],
              "foundation_version": "modernbert-base-lora16-v1", "r_max": 16, "hidden_size": h,
              "num_layers": cfg["num_hidden_layers"], "targets": {"attn.Wqkv": [h, 3*h], "attn.Wo": [h,h], "mlp.Wi": [h,2*intermediate], "mlp.Wo": [intermediate,h]}}
    write(root / "foundation.json", layout)
    return kit


def parser_check(root):
    snapshot = {"schema": 1, "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
    for key, url in (("family", "https://devnode.telegraphprotocol.com/api/wasm?resolve=" + FAMILY),
                     ("child", "https://devnode.telegraphprotocol.com/engine/v1/intents/" + FAMILY + "_GENERAL")):
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Translation-class-check/1.0"}), timeout=30) as response:
            raw = response.read((1 << 20) + 1)
            if response.status != 200 or len(raw) > 1 << 20:
                raise ValueError("primary class source status/size")
        snapshot[key] = {"url": url, "raw": raw.decode()}
    path = root / "class-snapshot.json"
    write(path, snapshot)
    command = [str(root / "packagecheck"), "--intent", FAMILY, "--child", FAMILY + "_GENERAL",
               "--class-snapshot", str(path), "--snapshot-sha256", sha(path),
               "--package", str(root / (FAMILY + ".safetensors")), "--sha256", PACKAGE_SHA,
               "--foundation", str(root / "foundation.json"), "--foundation-sha256", sha(root / "foundation.json"), "--out", str(root)]
    result = subprocess.run(command, capture_output=True, check=True, timeout=60)
    value = json.loads(result.stdout)
    if value["sha256"] != PACKAGE_SHA or value["keccak256"].removeprefix("0x") != PACKAGE_KECCAK or value["manifest"]["verification"] != {"stage1": True, "loaded": False}:
        raise ValueError("full parser proof mismatch")
    write(root / "package-manifest.json", value)
    return value


def replay(root):
    remote_only()
    _source, native = verify_assets()
    if not root.resolve().is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("replay root outside RUNNER_TEMP")
    prepare(root)
    package = parser_check(root)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(root / "foundation/tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    views = {style: rows(style) for style in ("raw", "json", "prose")}
    vectors = {style: [tokenizer.encode(f"### Task:\n{item['prompt']}\n\n### Answer:\n{item[side]}").ids for item in items for side in ("chosen", "rejected")] for style, items in views.items()}
    if any(not ids or len(ids) > 256 for batches in vectors.values() for ids in batches):
        raise ValueError("immutable audit distinction would be clipped")
    scope = helper_scope({})
    original_check = scope["check_response"]
    def exact_response(value, count, manifest):
        if value.get("adapter_hash") != PACKAGE_KECCAK:
            raise ValueError("literal bare adapter hash required")
        return original_check(value, count, manifest)
    scope["check_response"] = exact_response
    streams, negative = {}, 0
    try:
        with scope["server"](root, package, "first") as score:
            for style, batches in vectors.items():
                streams[style] = [[bit for start in range(0, 512, 4) for bit in score(batches[start:start+4])] for _ in range(2)]
                if streams[style][0] != streams[style][1]:
                    raise ValueError("reference bits changed on repetition")
                write(root / "runtime-score-bits.json", streams)
            for identity in ("foundation_hash", "tokenizer_hash", "adapter_hash"):
                try:
                    score(vectors["raw"][:1], {identity: "0" * 64})
                except ValueError:
                    negative += 1
                else:
                    raise ValueError("negative identity escaped verification")
        with scope["server"](root, package, "fresh") as score:
            for style, batches in vectors.items():
                fresh = [bit for start in range(0, 512, 4) for bit in score(batches[start:start+4])]
                if fresh != streams[style][0]:
                    raise ValueError("fresh-process reference bits changed")
        checks = json.loads(read_asset("source-checks.json"))
        shortcut = max(checks["shortcut_baselines"]["by_split"]["test"].values())
        formats = {}
        for style, items in views.items():
            metric = metrics(items, streams[style][0])
            metric["untrained_accuracy"] = native["untrained"][style]["pairwise_accuracy"]
            metric["shortcut_accuracy"] = shortcut
            metric["untrained_scope"] = "actual original native run, not new training or fitting"
            if (metric["accuracy"] < 0.80 or min(metric["per_label"].values()) < 0.70 or shortcut > 0.65
                    or metric["accuracy"] + 1e-9 < metric["untrained_accuracy"] + 0.05 or metric["accuracy"] + 1e-9 < shortcut + 0.10):
                raise ValueError("unchanged reference format/label/baseline quality gate failed: " + style)
            formats[style] = metric
        fixed = [{"tokens": ids, "score_bits": bit} for ids, bit in zip(vectors["raw"][:16], streams["raw"][0][:16])]
        report = {"schema_version": 1, "status": "REGISTRATION_READY", "family": FAMILY,
                  "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
                  "package_sha256": PACKAGE_SHA, "package_keccak": PACKAGE_KECCAK,
                  "foundation_sha256": FOUNDATION["foundation_lora_fp32.onnx"], "engine_sha256": FOUNDATION["foundation_engine.wasm"],
                  "tokenizer_sha256": FOUNDATION["tokenizer.json"], "serving_commit": os.environ["GITHUB_SHA"],
                  "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True,
                             "source_manifest_sha256": SOURCE_SHA, "independent_groups": 64},
                  "quality": {"passed": True, "audit_pairs": 256, "all_supported_formats_tested": True, "formats": formats},
                  "runtime": {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
                              "negative_identity_controls": negative, "vectors": fixed},
                  "replay": {"mode": "immutable_checkpoint_reference_runtime_replay", "training_run_id": TRAIN_RUN,
                             "training_commit": TRAIN_COMMIT, "new_training": False, "audit_used_for_new_fitting": False}}
        write(root / "registration-ready.json", report)
        write(root / "runtime-result.json", {"status": "PASS", "formats": formats, "new_training": False})
    except Exception as error:
        write(root / "runtime-error.json", {"status": "FAIL", "error_type": type(error).__name__, "error": str(error),
              "completed_formats": sorted(streams), "package_sha256": PACKAGE_SHA, "new_training": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    replay(parser.parse_args().root)
