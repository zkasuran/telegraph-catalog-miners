# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Preserve full source grants on successful and failed exports."""
import argparse
from pathlib import Path

from data import HERE, write_json
from licensing import SIDECARS, licence_metadata, preserve_artifact


def preserve(root):
    root.mkdir(parents=True, exist_ok=True)
    metadata = licence_metadata()
    for target in (root, root / "adapter", root / "initial", root / "checkpoint"):
        target.mkdir(exist_ok=True)
        for name in SIDECARS:
            (target / name).write_bytes((HERE / name).read_bytes())
        write_json(target / "licence-metadata.json", metadata)
        for path in sorted(target.glob("*.safetensors")) + sorted(target.glob("*.pt")):
            preserve_artifact(path)
    if (root / "data").is_dir():
        for name in ("CC-BY-4.0.txt", "NOTICE", "DATA-SOURCES.md"):
            (root / "data" / name).write_bytes((HERE / name).read_bytes())
        write_json(root / "data/LICENCE-SCOPE.json", {
            "source_text_and_original_title_associations": "CC-BY-4.0, not restricted by own SAND terms",
            "constructed_preferences_and_fitted_additions": "SAND only on own portions",
            "own_additions": metadata,
        })
    if not (root / "result.json").exists():
        write_json(root / "result.json", {"status": "INCOMPLETE", "phase": "not_started_or_interrupted",
                   "licence_metadata": metadata, "node_qualification": "not attempted"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    preserve(parser.parse_args().root)
