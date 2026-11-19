# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Pinned public reference inputs. Weight downloads are remote-only."""
import hashlib
import json
import sys
import tarfile
import time
import urllib.request

from data import HERE
from runtime import HASHES

KIT_COMMIT = "1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd"
KIT_SHA = "18d7dbe7425e977b576a70146f901a006b8e4abddbfd351276205faa7f287243"
BASE_COMMIT = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
RELEASE = "https://github.com/telegraphprotocol/telegraph-foundation/releases/download/modernbert-base-lora16-v1/"
BASE_FILES = {"model.safetensors": "340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9",
              "config.json": "1609d59e627c33eaed524b4f01e546d42e84190a079a5a5ded84b212c41c324f",
              "tokenizer_config.json": "3cd2017ff46d0a527e5d39cae39272eccfa1f19bb9f89b05d166aab2e38354e2",
              "special_tokens_map.json": "ea97ecdbcc73713039d8d64dbb05e3689495c96657fbd9a18f5bed381be81049"}


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for raw in iter(lambda: source.read(1 << 20), b""):
            h.update(raw)
    return h.hexdigest()


def fetch(url, target, expected, ceiling):
    if target.is_file() and not target.is_symlink() and target.stat().st_size <= ceiling and file_hash(target) == expected:
        return
    partial = target.with_name(target.name + ".partial")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started, total, h = time.monotonic(), 0, hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "email-security-reference/2"})
    try:
        with opener.open(request, timeout=45) as response, partial.open("wb") as output:
            if response.status != 200:
                raise ValueError("reference download status")
            while raw := response.read(1 << 20):
                total += len(raw)
                if total > ceiling or time.monotonic() - started > 600:
                    raise ValueError("reference download byte/time ceiling")
                h.update(raw)
                output.write(raw)
        if h.hexdigest() != expected:
            raise ValueError("reference download hash mismatch: " + target.name)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def prepare_kit(root):
    archive = root / "kit.tar.gz"
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + KIT_COMMIT,
          archive, KIT_SHA, 32 << 20)
    with tarfile.open(archive) as source:
        if sum(member.size for member in source.getmembers()) > 128 << 20:
            raise ValueError("source kit expansion ceiling")
        source.extractall(root, filter="data")
    return root / ("telegraph-wasm-baseline-" + KIT_COMMIT)


def prepare_reference(root):
    from remote import remote_only
    remote_only()
    foundation, base = root / "foundation", root / "base"
    foundation.mkdir(exist_ok=True)
    base.mkdir(exist_ok=True)
    for name, sha in HASHES.items():
        fetch(RELEASE + name, foundation / name, sha, 700 << 20)
    (base / "tokenizer.json").write_bytes((foundation / "tokenizer.json").read_bytes())
    for name, sha in BASE_FILES.items():
        fetch("https://huggingface.co/answerdotai/ModernBERT-base/resolve/" + BASE_COMMIT + "/" + name,
              base / name, sha, 700 << 20)
    kit = prepare_kit(root)
    sys.path.insert(0, str(kit))
    return foundation, base, kit


def source_binding():
    # No expected performance appears here. This file binds licensed source and
    # original grouped fixture bytes, not a claim about observed external DNS.
    value = json.loads((HERE / "source-manifest.json").read_bytes())
    for name, expected in value["licences"].items():
        if file_hash(HERE / name) != expected:
            raise ValueError("reviewed source grant bytes changed: " + name)
    if value.get("authorship") != "AI-assisted constructed protocol fixtures" or value.get("external_dns_truth") is not False:
        raise ValueError("source was relabelled as external DNS truth")
    return value
