# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Retain failed exports with all upstream grants, without neural imports."""
import argparse
from pathlib import Path

from contracts import FAMILY, sha, write
from licensing import licence_metadata, preserve_artifact


def preserve(root):
    root.mkdir(parents=True, exist_ok=True)
    write(root / "TERMS.json", licence_metadata())
    records = []
    for directory in (root / "initial", root / "checkpoint", root / "selected"):
        if not directory.is_dir():
            continue
        write(directory / "TERMS.json", licence_metadata())
        for path in directory.iterdir():
            if path.is_file() and path.suffix in (".safetensors", ".pt"):
                preserve_artifact(path)
                records.append({"path": str(path.relative_to(root)), "sha256": sha(path.read_bytes())})
    write(root / "preserved.json", {"family": FAMILY, "records": records, "terms_sha256": sha((root / "TERMS.json").read_bytes())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    preserve(parser.parse_args().root)
