# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded public results allowlist with the same licence boundary on failure."""
import argparse
import json
import shutil
from pathlib import Path

from bootstrap import dump
from checkpoint import licence_metadata, preserve_artifact
from source import HERE, sha

MAX_TOTAL = 384 << 20


def preserve(root):
    root.mkdir(parents=True, exist_ok=True)
    output = root / "public-results"
    output.mkdir(exist_ok=True)
    for name in ("LICENSE", "NOTICE", "UPSTREAM-APACHE-2.0.txt", "KIT-LICENSE", "CC0.txt", "DATA-SOURCES.md"):
        shutil.copyfile(HERE / name, output / name)
    licence_metadata()
    paths = list(root.glob("*.json")) + list(root.glob("*.log")) + list(root.glob("*.safetensors"))
    for directory in ("initial", "selected", "checkpoint", "selected-epoch-1", "selected-epoch-2", "selected-epoch-3"):
        base = root / directory
        if base.exists():
            paths.extend(p for p in base.iterdir() if p.is_file() and p.suffix in {".json", ".safetensors", ".pt"})
    if (root / "class").is_dir():
        paths.extend((root / "class").glob("*.json"))
    total, artifacts, refused = 0, [], []
    for source in sorted(set(paths)):
        if source.is_symlink() or source.stat().st_size > 64 << 20:
            refused.append(source.name)
            continue
        try:
            if source.suffix in {".safetensors", ".pt"}:
                preserve_artifact(source)
            if source.suffix == ".json":
                json.loads(source.read_bytes())
        except Exception as exc:
            refused.append({"file": source.name, "reason": type(exc).__name__})
            continue
        raw = source.read_bytes()
        total += len(raw)
        if total > MAX_TOTAL:
            raise ValueError("public result artifact ceiling")
        relative = source.relative_to(root)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        artifacts.append({"path": str(relative), "sha256": sha(raw), "bytes": len(raw)})
    dump(output / "public-artifact-manifest.json", {"schema_version": 1, "files": artifacts, "refused_files": refused,
         "licence_boundary_verified": True, "foundation_weights_uploaded": False, "provider_credentials_uploaded": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    preserve(parser.parse_args().root)
