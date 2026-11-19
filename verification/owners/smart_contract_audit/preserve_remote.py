# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Always-run failure preservation. Pure byte stamping, no model operations."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    root = parser.parse_args().root
    here = Path(__file__).resolve().parent
    checkpoint = root / "adapter" / "adapter_model.safetensors"
    if not checkpoint.exists():
        print("No saved candidate checkpoint exists yet")
        return
    from byte_stamp import stamp
    licences = json.loads((here / "licences.json").read_text())
    stamped = stamp(checkpoint.read_bytes(), licences)
    (root / "checkpoint.safetensors").write_bytes(stamped)
    (root / "checkpoint-config.json").write_bytes((root / "adapter" / "adapter_config.json").read_bytes())
    print("Saved best checkpoint retained with upstream grants and SAND terms in bytes")


if __name__ == "__main__":
    main()
