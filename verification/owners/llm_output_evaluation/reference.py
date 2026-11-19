# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pinned public reference downloads. Entry point guards precede model work."""
import json
import os
import sys
import tarfile
import time
import urllib.request
import hashlib
from source_gate import HERE
from common import FOUNDATION, FOUNDATION_RELEASE


def fetch(url, target, sha, ceiling=1 << 30):
    if target.exists() and target.stat().st_size <= ceiling and hashlib.sha256(target.read_bytes()).hexdigest() == sha:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "Telegraph-owner-reference/1"})
    start, total, digest = time.monotonic(), 0, hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=60) as source, partial.open("wb") as out:
            while chunk := source.read(1 << 20):
                total += len(chunk)
                if total > ceiling or time.monotonic() - start > 600:
                    raise ValueError("reference download ceiling")
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != sha:
            raise ValueError("reference SHA-256 mismatch")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def prepare(root):
    pins = json.loads((HERE / "reference_inputs.json").read_text())
    foundation, base = root / "foundation", root / "base"
    for name, sha in FOUNDATION.items():
        fetch(FOUNDATION_RELEASE + name, foundation / name, sha)
    base.mkdir(exist_ok=True)
    (base / "tokenizer.json").write_bytes((foundation / "tokenizer.json").read_bytes())
    for name, sha in pins["base"]["files"].items():
        fetch(f"https://huggingface.co/{pins['base']['id']}/resolve/{pins['base']['revision']}/{name}", base / name, sha)
    archive = root / "kit.tar.gz"
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + pins["kit"]["revision"], archive, pins["kit"]["sha256"], 32 << 20)
    with tarfile.open(archive) as stream:
        if sum(m.size for m in stream.getmembers()) > 128 << 20:
            raise ValueError("kit expansion ceiling")
        stream.extractall(root, filter="data")
    kit = root / ("telegraph-wasm-baseline-" + pins["kit"]["revision"])
    sys.path.insert(0, str(kit))
    os.environ["HF_HUB_OFFLINE"] = "1"
    return foundation, base, kit
