# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Standalone pinned preparation. All model downloads require remote authority."""
import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path
from urllib.request import Request, urlopen

from policy import remote_only

HERE = Path(__file__).resolve().parent
RELEASE = "https://github.com/telegraphprotocol/telegraph-foundation/releases/download/modernbert-base-lora16-v1/"


def fetch(url, target, expected, ceiling=1 << 30):
    remote_only()
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
        return
    partial = target.with_suffix(target.suffix + ".partial")
    digest, total, start = hashlib.sha256(), 0, time.monotonic()
    with urlopen(Request(url, headers={"User-Agent": "smart-audit-owner/2"}), timeout=60) as source, partial.open("wb") as output:
        while chunk := source.read(1 << 20):
            total += len(chunk)
            if total > ceiling or time.monotonic() - start > 600:
                raise ValueError("download ceiling")
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("download identity mismatch")
    partial.replace(target)


def prepare(root, training=True):
    remote_only()
    sources = json.loads((HERE / "sources.json").read_text())
    hashes = sources["foundation"]
    foundation, base = root / "foundation", root / "base"
    foundation.mkdir(exist_ok=True)
    base.mkdir(exist_ok=True)
    for name, expected in hashes.items():
        fetch(RELEASE + name, foundation / name, expected)
    source = sources["base"]
    for name, expected in source["files"].items():
        if not training and name != "config.json":
            continue
        fetch(f"https://huggingface.co/{source['id']}/resolve/{source['revision']}/{name}", base / name, expected)
    if training:
        (base / "tokenizer.json").write_bytes((foundation / "tokenizer.json").read_bytes())
    pin = sources["kit"]
    archive = root / "kit.tar.gz"
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + pin["revision"], archive, pin["sha256"], 32 << 20)
    with tarfile.open(archive) as tar:
        if sum(m.size for m in tar.getmembers()) > 128 << 20:
            raise ValueError("kit ceiling")
        tar.extractall(root, filter="data")
    kit = root / ("telegraph-wasm-baseline-" + pin["revision"])
    sys.path.insert(0, str(kit))
    os.environ["HF_HUB_OFFLINE"] = "1"
    return foundation, base, kit, hashes, sources


def bound_data(root):
    from data import check, sha
    rows = json.loads((root / "data/inventory.json").read_text())
    check(rows)
    expected = json.loads((HERE / "data-bindings.json").read_text())
    if expected.get("retired_exposed_audit_sha256") == expected["files"]["audit"]["sha256"]:
        raise ValueError("exposed original source audit cannot be reused for qualification")
    if expected.get("retired_audit_used_for_fitting_or_new_test") is not False:
        raise ValueError("explicit retired-audit exclusion required")
    for split, entry in expected["files"].items():
        if sha((root / "data" / (split + ".jsonl")).read_bytes()) != entry["sha256"]:
            raise ValueError("split identity changed")
    if sha((root / "data/inventory.json").read_bytes()) != expected["inventory_sha256"]:
        raise ValueError("inventory identity changed")
    if expected["independent_groups"] < 32 or expected["files"]["audit"]["rows"] < 64:
        raise ValueError("source independent group or audit pair gate")
    return rows, expected
