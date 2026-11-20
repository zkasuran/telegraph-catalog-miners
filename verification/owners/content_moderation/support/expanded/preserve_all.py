# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only preservation after success, timeout or model failure."""
import argparse
import json
from pathlib import Path

from core import load_shared, sha, write


def preserve(root):
    _, _, module = load_shared()
    licences = json.loads((Path(__file__).parent.parent / "training" / "licences.json").read_text())
    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    retained = []
    for directory in sorted((root / "best").glob("*")):
        tensor, config = directory / "adapter_model.safetensors", directory / "adapter_config.json"
        if not tensor.is_file():
            continue
        target = evidence / (directory.name + "-best.safetensors")
        target.write_bytes(module.stamp(tensor.read_bytes(), licences))
        if config.is_file():
            (evidence / (directory.name + "-best-config.json")).write_bytes(config.read_bytes())
        retained.append({"file": target.name, "sha256": sha(target.read_bytes())})
    write(evidence / "preservation.json", {"best_checkpoints": retained,
          "epoch_snapshots": [{"file": p.name, "sha256": sha(p.read_bytes())} for p in sorted(evidence.glob("*.safetensors"))],
          "upstream_grants_embedded": True})
    write(evidence / "licences.json", licences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    preserve(parser.parse_args().root)
