# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Attach full source terms to every result, delta and failed checkpoint."""
import argparse
import json
from pathlib import Path

from licensing import SIDECARS, licence_metadata, preserve_artifact

HERE = Path(__file__).resolve().parent


def preserve(root):
    root.mkdir(parents=True, exist_ok=True)
    metadata = licence_metadata()
    for target in (root, root / "adapter", root / "initial", root / "checkpoint"):
        target.mkdir(exist_ok=True)
        for name in SIDECARS:
            (target / name).write_bytes((HERE / name).read_bytes())
        (target / "licence-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        for path in sorted(target.glob("*.safetensors")) + sorted(target.glob("*.pt")):
            preserve_artifact(path)
    # Original source bytes stay under their own licence. Own fitted additions
    # in this directory receive a separate sidecar, without changing freeze bytes.
    if (root / "data").is_dir():
        (root / "data/OWN-ADDITIONS-LICENSE").write_bytes((HERE / "LICENSE").read_bytes())
        (root / "data/LICENCE-SCOPE.json").write_text(json.dumps({"own_additions": metadata,
            "source_text_and_original_annotations": "CC0-1.0 structured facts, not restricted by own SAND terms",
            "frozen_diagnostic_coefficients": "SAND on own additions only; original source facts retain CC0-1.0"}, indent=2) + "\n")
    result = root / "result.json"
    if not result.exists():
        result.write_text(json.dumps({"status": "INCOMPLETE", "phase": "interrupted_or_not_started", "licence_metadata": metadata,
                                     "runtime_integration": "INCOMPLETE", "node_qualification": "not attempted"}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    preserve(parser.parse_args().root)
