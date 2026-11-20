# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded reference preparation. Source fetching never loads model weights."""
import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

from data import HERE, fetch
from runtime_check import RELEASE_HASHES, model_remote_only

RELEASE = "https://github.com/telegraphprotocol/telegraph-foundation/releases/download/modernbert-base-lora16-v1/"


def prepare_kit(root):
    pin = json.loads((HERE / "reference-pins.json").read_text())["kit"]
    archive = root / "kit.tar.gz"
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + pin["revision"],
          archive, pin["sha256"], 32 << 20)
    kit = root / ("telegraph-wasm-baseline-" + pin["revision"])
    with tarfile.open(archive) as source:
        members = source.getmembers()
        if len(members) > 10000 or sum(m.size for m in members) > 128 << 20:
            raise ValueError("source kit expansion ceiling")
        if any(not (m.isfile() or m.isdir()) or not (root / m.name).resolve().is_relative_to(kit.resolve()) for m in members):
            raise ValueError("source kit member escape or link")
        source.extractall(root, filter="data")
    sys.path.insert(0, str(kit))
    return kit


def prepare_reference(root):
    model_remote_only()
    model, base = root / "foundation", root / "base"
    model.mkdir(exist_ok=True)
    base.mkdir(exist_ok=True)
    for name, sha in RELEASE_HASHES.items():
        fetch(RELEASE + name, model / name, sha)
    (base / "tokenizer.json").write_bytes((model / "tokenizer.json").read_bytes())
    spec = json.loads((HERE / "reference-pins.json").read_text())["base"]
    for name, sha in spec["files"].items():
        fetch(f"https://huggingface.co/{spec['id']}/resolve/{spec['revision']}/{name}", base / name, sha)
    os.environ["HF_HUB_OFFLINE"] = "1"
    return model, base, root / ("telegraph-wasm-baseline-" + json.loads((HERE / "reference-pins.json").read_text())["kit"]["revision"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit-only", action="store_true", required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    print(prepare_kit(args.root))
