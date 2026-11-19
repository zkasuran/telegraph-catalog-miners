# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Retain failed or successful best checkpoints with the same source licence boundary."""
import argparse
import json
import struct
from pathlib import Path

from terms_policy import licences_for


def stamp(raw, licences):
    if len(raw) < 9 or len(raw) > 64 << 20:
        raise ValueError("checkpoint byte ceiling")
    size = struct.unpack("<Q", raw[:8])[0]
    if not 0 < size <= 1 << 20 or 8 + size > len(raw):
        raise ValueError("checkpoint header ceiling")
    header = json.loads(raw[8:8 + size])
    before = {k: v for k, v in header.items() if k != "__metadata__"}
    metadata = header.setdefault("__metadata__", {})
    for key, value in licences.items():
        metadata["x-terms-" + key] = value
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    if len(encoded) > 1 << 20:
        raise ValueError("stamped checkpoint header ceiling")
    result = struct.pack("<Q", len(encoded)) + encoded + raw[8 + size:]
    if before != {k: v for k, v in json.loads(encoded).items() if k != "__metadata__"} or result[8 + len(encoded):] != raw[8 + size:]:
        raise ValueError("checkpoint stamp changed tensor bytes")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.root / "adapter" / "adapter_model.safetensors"
    if not checkpoint.exists():
        print("No saved checkpoint. No model or result was invented.")
        return
    licences = licences_for(args.root)
    raw = stamp(checkpoint.read_bytes(), licences)
    (args.root / "checkpoint.safetensors").write_bytes(raw)
    cfg = checkpoint.parent / "adapter_config.json"
    if cfg.exists():
        (args.root / "checkpoint-config.json").write_bytes(cfg.read_bytes())
    print("Best saved checkpoint retained with original tensor bytes and full upstream/addition notices.")


if __name__ == "__main__":
    main()
