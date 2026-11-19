# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Standalone hash-pinned reference preparation, guarded before any weight fetch."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import urllib.request

HERE = Path(__file__).resolve().parent
SOURCES = json.loads((HERE / "model-sources.json").read_text())
FOUNDATION = json.loads((HERE / "contract.json").read_text())["foundation"]
RELEASE = "https://github.com/telegraphprotocol/telegraph-foundation/releases/download/modernbert-base-lora16-v1/"


def remote_only():
    if (os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_REPOSITORY") != "zkasuran/telegraph-catalog-miners"
            or os.environ.get("GITHUB_WORKFLOW") != "Weather archive evaluator experiment"
            or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
            or not os.environ.get("RUNNER_TEMP")):
        raise RuntimeError("model operations require the controller's bounded public CPU workflow")


def fetch(url, target, expected, ceiling=1 << 30):
    remote_only()
    allowed = (RELEASE, "https://huggingface.co/answerdotai/ModernBERT-base/resolve/",
               "https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/",
               "https://raw.githubusercontent.com/zkasuran/telegraph-catalog-miners/")
    if not url.startswith(allowed) or "?" in url:
        raise ValueError("reference fetch outside bounded immutable source namespace")
    if target.exists():
        raw = target.read_bytes()
        if len(raw) <= ceiling and hashlib.sha256(raw).hexdigest() == expected:
            return
        raise ValueError("existing source bytes disagree with immutable identity")
    request = urllib.request.Request(url, headers={"User-Agent": "Weather-owner-reference/1.0"})
    partial = target.with_suffix(target.suffix + ".partial")
    deadline = time.monotonic() + 600
    total, digest = 0, hashlib.sha256()
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as destination:
        while chunk := response.read(1 << 20):
            total += len(chunk)
            if total > ceiling or time.monotonic() > deadline:
                raise ValueError("reference download byte/time ceiling")
            digest.update(chunk)
            destination.write(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("reference SHA-256 mismatch")
    partial.replace(target)


def source_kit(root):
    remote_only()
    pin = SOURCES["kit"]
    archive = root / "kit.tar.gz"
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + pin["revision"],
          archive, pin["sha256"], 32 << 20)
    kit = root / ("telegraph-wasm-baseline-" + pin["revision"])
    if not kit.exists():
        with tarfile.open(archive) as source:
            members = source.getmembers()
            if len(members) > 5000 or sum(m.size for m in members) > 128 << 20:
                raise ValueError("source kit member/expansion ceiling")
            source.extractall(root, filter="data")
    return kit


def foundation(root):
    remote_only()
    model = root / "foundation"
    model.mkdir(exist_ok=True)
    for name, digest in FOUNDATION.items():
        fetch(RELEASE + name, model / name, digest)
    return model


def prepare(root):
    remote_only()
    model, base = foundation(root), root / "base"
    base.mkdir(exist_ok=True)
    (base / "tokenizer.json").write_bytes((model / "tokenizer.json").read_bytes())
    spec = SOURCES["base"]
    for name, digest in spec["files"].items():
        fetch(f"https://huggingface.co/{spec['id']}/resolve/{spec['revision']}/{name}", base / name, digest)
    kit = source_kit(root)
    sys.path.insert(0, str(kit))
    os.environ["HF_HUB_OFFLINE"] = "1"
    return model, base, kit


def compile_tools(root, kit):
    remote_only()
    for name, source in (("adapterexec", "./cmd/adapterexec"), ("stagecheck", str(HERE / "stagecheck.go"))):
        subprocess.run(["go", "build", "-mod=readonly", "-trimpath", "-o", str(root / name), source],
                       cwd=kit, check=True, timeout=600)


def validate_package(root, package, output_name):
    remote_only()
    from live_class import capture_classes
    from build import sha, write_json
    capture_classes(root)
    command = [str(root / "stagecheck"), "--intent", "WEATHER_FORECAST_VERIFY",
               "--child", "WEATHER_FORECAST_VERIFY_GENERAL", "--expected-class", "judgment",
               "--package", str(package), "--sha256", sha(package.read_bytes()),
               "--foundation", str(root / "foundation.json"), "--foundation-sha256", sha((root / "foundation.json").read_bytes()),
               "--class-snapshot", str(root / "class-snapshot.json"),
               "--snapshot-sha256", sha((root / "class-snapshot.json").read_bytes()), "--out", str(root)]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    report = json.loads(result.stdout)
    if report["sha256"] != sha(package.read_bytes()) or report["manifest"]["verification"] != {"stage1": True, "loaded": False}:
        raise ValueError("full parser did not bind exact package or run source-only stage")
    write_json(root / output_name, report)
    return report
