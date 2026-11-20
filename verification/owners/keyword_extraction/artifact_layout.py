# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Stage bounded public metadata under one artifact root, never a private key."""
import argparse
import hashlib
import json
from pathlib import Path


def validate_names(names, kind, complete=False):
    if kind not in {"train", "audit"} or len(names) != len(set(names)):
        raise ValueError("artifact kind or duplicate member names")
    for name in names:
        if Path(name).name != name or name.startswith(".") or not name.endswith((".json", ".log", ".safetensors")):
            raise ValueError("artifact must use safe literal root filenames")
        if name.lower().startswith(("forward-key", "private-key", "ssh-key", "signer", "wallet")):
            raise ValueError("private key or unexpected artifact member")
    required = {"candidate.json", "foundation.json", "frozen-baselines.json", "KEYWORD_EXTRACTION.safetensors", "KEYWORD_EXTRACTION_INITIAL.safetensors"} if kind == "train" else {"registration-ready.json", "audit-result.json"}
    if complete and not required <= set(names):
        raise ValueError("complete artifact lacks its literal consumer contract")
    return True


def stage(root, kind, dependency=None, licences=None):
    root.mkdir(parents=True, exist_ok=True)
    for source, name in ((dependency, "dependency-audit.json"), (licences, "licences.json")):
        if source is None or not source.is_file():
            continue  # Early failures still retain real existing diagnostics.
        if source.is_symlink() or source.stat().st_size > 1 << 20:
            raise ValueError("public metadata size or symlink refused")
        raw = source.read_bytes()
        json.loads(raw)
        (root / name).write_bytes(raw)
    files = [p for p in root.iterdir() if p.is_file() and p.name != "artifact-layout.json" and p.suffix in {".json", ".log", ".safetensors"}]
    validate_names([p.name for p in files], kind)
    records = []
    for path in files:
        if path.is_symlink() or path.stat().st_size > 64 << 20:
            raise ValueError("artifact member size or symlink refused")
        records.append({"name": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (root / "artifact-layout.json").write_text(json.dumps({"schema_version": 1, "family": "KEYWORD_EXTRACTION", "kind": kind,
                                                          "files": sorted(records, key=lambda r: r["name"]), "scope": "Literal root filenames, metadata only. No model quality or canonical status inferred."}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=("train", "audit"), required=True)
    parser.add_argument("--dependency", type=Path)
    parser.add_argument("--licences", type=Path)
    args = parser.parse_args()
    stage(args.root, args.kind, args.dependency, args.licences)


if __name__ == "__main__":
    main()
