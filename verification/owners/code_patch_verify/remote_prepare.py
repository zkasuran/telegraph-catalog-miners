# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Pinned remote source/base preparation. Refuses local and unattested work."""
import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

from remote import caller_contract, remote_only


def fetch(url, path, expected, ceiling):
    total, digest = 0, hashlib.sha256()
    with urllib.request.urlopen(url, timeout=30) as source, path.open("wb") as target:
        while chunk := source.read(1 << 20):
            total += len(chunk)
            if total > ceiling:
                raise ValueError("remote source byte ceiling")
            digest.update(chunk)
            target.write(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("remote source hash mismatch")


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    args = parser.parse_args()
    caller_contract(args.attestation)
    pins = json.loads((args.shared / "sources.json").read_text())
    root = args.root
    root.mkdir(exist_ok=True)
    base = root / "base"
    base.mkdir(exist_ok=True)
    for name, sha in pins["base"]["files"].items():
        fetch("https://huggingface.co/" + pins["base"]["id"] + "/resolve/" + pins["base"]["revision"] + "/" + name,
              base / name, sha, 1 << 30)
    fetch("https://github.com/telegraphprotocol/telegraph-foundation/releases/download/modernbert-base-lora16-v1/tokenizer.json",
          base / "tokenizer.json", "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456", 8 << 20)
    archive = root / "kit.tar.gz"
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + pins["kit"]["revision"],
          archive, pins["kit"]["sha256"], 32 << 20)
    with tarfile.open(archive) as source:
        if sum(member.size for member in source.getmembers()) > 128 << 20:
            raise ValueError("source archive expansion ceiling")
        source.extractall(root, filter="data")
    print("Prepared pinned base and source kit on controller CPU runner")


if __name__ == "__main__":
    main()
