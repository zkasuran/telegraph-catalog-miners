# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Licence failed and successful checkpoints before a controller artifact export."""
import argparse
from pathlib import Path

from data import digest, write_json
from remote import licence_metadata, stamp_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(exist_ok=True)
    terms = licence_metadata(args.shared)
    write_json(args.root / "licences.json", terms)
    files = []
    for pattern in ("*.safetensors", "adapter/*.safetensors", "adapter/*.pt", "checkpoint/*.safetensors", "checkpoint/*.pt"):
        for path in sorted(args.root.glob(pattern)):
            stamp_checkpoint(path, terms)
            files.append({"file": str(path.relative_to(args.root)), "sha256": digest(path.read_bytes())})
    write_json(args.root / "export-bindings.json", {"files": files, "terms_embedded": True,
               "node_qualification": "not_attempted", "runtime_verified": False})
    print("Verified export terms on " + str(len(files)) + " checkpoint files")


if __name__ == "__main__":
    main()
