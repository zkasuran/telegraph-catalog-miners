# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pinned public CPU inputs and actual full Go package parser. Remote only."""
import subprocess
import sys
import tarfile
from pathlib import Path

from contracts import FAMILY, FOUNDATION, HERE, file_sha, remote_only, strict_json, write_json
from data import download

RELEASE = "https://github.com/telegraphprotocol/telegraph-foundation/releases/download/modernbert-base-lora16-v1/"


def prepare(root, weights=False):
    remote_only()
    root.mkdir(parents=True, exist_ok=True)
    pins = strict_json((HERE / "runtime-pins.json").read_bytes())
    foundation, base = root / "foundation", root / "base"
    foundation.mkdir(exist_ok=True)
    base.mkdir(exist_ok=True)
    for name, expected in FOUNDATION.items():
        download(RELEASE + name, foundation / name, expected, 1536 << 20)
        if file_sha(foundation / name) != expected:
            raise ValueError("actual foundation input identity mismatch")
    spec = pins["base"]
    for name, expected in spec["files"].items():
        if name == "model.safetensors" and not weights:
            continue
        download(f"https://huggingface.co/{spec['id']}/resolve/{spec['revision']}/{name}", base / name, expected, 768 << 20)
    (base / "tokenizer.json").write_bytes((foundation / "tokenizer.json").read_bytes())
    archive = root / "kit.tar.gz"
    commit = pins["kit"]["revision"]
    download("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/"+commit,
             archive, pins["kit"]["sha256"], 32 << 20)
    kit = root / ("telegraph-wasm-baseline-" + commit)
    if not kit.exists():
        with tarfile.open(archive) as source:
            if sum(item.size for item in source.getmembers()) > 128 << 20:
                raise ValueError("reference kit extraction ceiling")
            source.extractall(root, filter="data")
    sys.path.insert(0, str(kit))
    return foundation, base, kit


def compile_runtime(root, kit):
    remote_only()
    for name in ("adapterexec", "adapterbuild"):
        subprocess.run(["go", "build", "-trimpath", "-o", str(root/name), "./cmd/"+name], cwd=kit, check=True, timeout=600)
    (kit / "owner-stagecheck.go").write_bytes((HERE / "stagecheck.go").read_bytes())
    (kit / "owner-stagecheck_test.go").write_bytes((HERE / "stagecheck_test.go").read_bytes())
    subprocess.run(["go", "build", "-trimpath", "-o", str(root/"stagecheck"), "owner-stagecheck.go"],
                   cwd=kit, check=True, timeout=600)
    tested = subprocess.run(["go", "test", "owner-stagecheck.go", "owner-stagecheck_test.go", "-v"],
                            cwd=kit, check=True, capture_output=True, text=True, timeout=180)
    (root/"full-parser-tests.log").write_text(tested.stdout+tested.stderr)


def full_parser(root, package, foundation_spec):
    remote_only()
    snapshot = root / "class-snapshot.json"
    subprocess.run([sys.executable, str(HERE/"live.py"), "--out", str(snapshot)], check=True, timeout=90)
    raw = strict_json(snapshot.read_bytes())
    envelope = root / "class-envelope.json"
    write_json(envelope, {"schema": 1, "fetched_at": raw["fetched_at"],
                         "family": {"url": raw["family_url"], "raw": raw["family_raw"]},
                         "child": {"url": raw["child_url"], "raw": raw["child_raw"]}})
    process = subprocess.run([str(root/"stagecheck"), "--intent", FAMILY, "--child", FAMILY+"_GENERAL",
        "--expected-class", "judgment", "--class-snapshot", str(envelope), "--snapshot-sha256", file_sha(envelope),
        "--package", str(package), "--sha256", file_sha(package), "--foundation", str(foundation_spec),
        "--foundation-sha256", file_sha(foundation_spec), "--out", str(root)], capture_output=True, text=True, timeout=90, check=True)
    manifest = strict_json(process.stdout)
    if manifest["sha256"] != file_sha(package) or manifest["manifest"]["verification"] != {"stage1": True, "loaded": False}:
        raise ValueError("full parser did not verify exact package")
    write_json(root / (package.stem + "-parser.json"), manifest)
    return manifest
