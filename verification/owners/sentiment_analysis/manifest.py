# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Explicit public source/model manifest for the exact owned namespace."""
import argparse
from pathlib import Path

from contracts import FAMILY, HERE, file_sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checkout = HERE.parents[2]
    output = args.out.resolve()
    if not output.is_relative_to(checkout):
        raise ValueError("manifest output must remain in the owned checkout")
    workflow = checkout / ".github/workflows/evaluator-owner-sentiment_analysis.yml"
    files = []
    for path in sorted([*HERE.rglob("*"), workflow]):
        if not path.is_file() or path.is_symlink() or path.name == "original.safetensors":
            continue
        relative = path.relative_to(checkout).as_posix()
        if any(part.startswith(".") for part in path.relative_to(HERE).parts) if path != workflow else False:
            continue
        if path.name not in {"LICENSE", "NOTICE"} and path.suffix not in {".py", ".go", ".json", ".sh", ".md", ".lock", ".safetensors", ".yml"}:
            continue
        if path.stat().st_size > 64 << 20:
            raise ValueError("public artifact exceeds the byte ceiling")
        files.append({"source": relative, "target": relative, "sha256": file_sha(path), "bytes": path.stat().st_size})
    if not 1 <= len(files) <= 250:
        raise ValueError("public artifact inventory count")
    write_json(output, {"family": FAMILY, "run_id": args.run_id, "files": files})
    print("OWNED_PUBLIC_MANIFEST_FILES " + str(len(files)))


if __name__ == "__main__":
    main()
