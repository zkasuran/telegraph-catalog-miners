# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Keep the best saved checkpoint with terms embedded in its tensor container."""
import argparse
import json
import struct
from pathlib import Path


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
    new_header = json.dumps(header, separators=(",", ":")).encode()
    new_header += b" " * (-len(new_header) % 8)
    if len(new_header) > 1 << 20:
        raise ValueError("stamped checkpoint header ceiling")
    output = struct.pack("<Q", len(new_header)) + new_header + raw[8 + size:]
    parsed = json.loads(output[8:8 + len(new_header)])
    if before != {k: v for k, v in parsed.items() if k != "__metadata__"} or output[8 + len(new_header):] != raw[8 + size:]:
        raise ValueError("stamp changed checkpoint tensors")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    root = parser.parse_args().root
    checkpoint = root / "adapter" / "adapter_model.safetensors"
    if not checkpoint.exists():
        print("No saved epoch checkpoint yet")
        return
    licences = json.loads((Path(__file__).resolve().parent / "licences.json").read_text())
    output = root / "checkpoint.safetensors"
    partial = output.with_suffix(".stamped")
    partial.write_bytes(stamp(checkpoint.read_bytes(), licences))
    partial.replace(output)
    config = root / "adapter" / "adapter_config.json"
    (root / "checkpoint-config.json").write_bytes(config.read_bytes())
    print("Best saved checkpoint retained with byte-identical tensors and embedded terms")


if __name__ == "__main__":
    main()
