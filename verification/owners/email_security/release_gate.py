# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Source-only checks for exact licensed remote-qualified public package bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from licensing import licence_metadata
from ready import verify_ready


def checked_bytes(raw, report, terms):
    verify_ready(report)
    if len(raw) > 64 << 20 or len(raw) < 9 or len(raw) != report.get("package_bytes"):
        raise ValueError("exact qualified package byte count required")
    if hashlib.sha256(raw).hexdigest() != report["package_sha256"]:
        raise ValueError("package differs from remote-qualified bytes")
    header_bytes = struct.unpack("<Q", raw[:8])[0]
    if not 0 < header_bytes <= 1 << 20 or 8 + header_bytes >= len(raw):
        raise ValueError("bounded package header required")
    header = json.loads(raw[8:8 + header_bytes])
    if not isinstance(header, dict) or not isinstance(header.get("__metadata__"), dict):
        raise ValueError("licensed package metadata required")
    metadata = header["__metadata__"]
    if any(metadata.get(key) != value for key, value in terms.items()):
        raise ValueError("complete retained upstream grants and own terms required")
    return {"package_sha256": report["package_sha256"], "package_bytes": len(raw),
            "run_id": report["run_id"], "commit": report["commit"], "neural_loaded": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "packages")
    args = parser.parse_args()
    if not args.root.is_dir() or args.root.is_symlink():
        raise ValueError("public qualified package inventory required")
    directories = sorted(args.root.iterdir())
    if not 1 <= len(directories) <= 6:
        raise ValueError("bounded complete package inventory required")
    for directory in directories:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("package inventory contains unsafe path")
        files = {path.name for path in directory.iterdir()}
        if files != {"EMAIL_SECURITY.safetensors", "registration-ready.json"}:
            raise ValueError("public package directory has missing or unexpected files")
        package, report_path = directory / "EMAIL_SECURITY.safetensors", directory / "registration-ready.json"
        if package.is_symlink() or report_path.is_symlink() or package.stat().st_size > 64 << 20 or report_path.stat().st_size > 1 << 20:
            raise ValueError("package path or byte ceiling")
        result = checked_bytes(package.read_bytes(), json.loads(report_path.read_bytes()), licence_metadata())
        print(json.dumps(result))
    print("PASS complete licensed qualified-package inventory, source-only; no neural execution")


if __name__ == "__main__":
    main()
