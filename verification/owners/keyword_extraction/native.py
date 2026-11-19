# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Standalone pinned download/build support. Neural assets are remote-only."""
import hashlib
import json
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = json.loads((HERE / "model-sources.json").read_text())
RELEASE = "https://github.com/telegraphprotocol/telegraph-foundation/releases/download/modernbert-base-lora16-v1/"
FOUNDATION = {
    "foundation_lora_fp32.onnx": "0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6",
    "foundation_engine.wasm": "77643d6e1176ab8744801cef93ccb844d0136ed32b8daa865d541f37d13292de",
    "tokenizer.json": "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456",
}


def fetch(url, target, expected, ceiling=700 << 20):
    if target.is_file() and target.stat().st_size <= ceiling:
        h = hashlib.sha256()
        with target.open("rb") as cached:
            for chunk in iter(lambda: cached.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() == expected:
            return
    request = urllib.request.Request(url, headers={"User-Agent": "Telegraph-keyword-owner/1.0"})
    partial = target.with_suffix(target.suffix + ".partial")
    total, started, h = 0, time.monotonic(), hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=45) as response, partial.open("wb") as out:
            while chunk := response.read(1 << 20):
                total += len(chunk)
                if total > ceiling or time.monotonic() - started > 600:
                    raise ValueError("pinned download exceeds byte or time ceiling")
                h.update(chunk)
                out.write(chunk)
        if h.hexdigest() != expected:
            raise ValueError("pinned asset hash mismatch")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def prepare(root):
    from remote import remote_only
    remote_only()
    model, base = root / "foundation", root / "base"
    model.mkdir(exist_ok=True)
    base.mkdir(exist_ok=True)
    for name, digest in FOUNDATION.items():
        fetch(RELEASE + name, model / name, digest)
    (base / "tokenizer.json").write_bytes((model / "tokenizer.json").read_bytes())
    spec = SOURCES["base"]
    for name, digest in spec["files"].items():
        fetch(f"https://huggingface.co/{spec['id']}/resolve/{spec['revision']}/{name}", base / name, digest)
    commit = SOURCES["kit"]["revision"]
    archive = root / "kit.tar.gz"
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + commit,
          archive, SOURCES["kit"]["sha256"], 32 << 20)
    with tarfile.open(archive) as tar:
        if sum(m.size for m in tar.getmembers()) > 128 << 20:
            raise ValueError("reference kit extraction ceiling exceeded")
        tar.extractall(root, filter="data")
    kit = root / ("telegraph-wasm-baseline-" + commit)
    sys.path.insert(0, str(kit))
    os.environ["HF_HUB_OFFLINE"] = "1"
    return model, base, kit
