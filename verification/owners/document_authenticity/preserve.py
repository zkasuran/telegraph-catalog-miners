# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Embed retained terms without changing any successful or failed tensor bytes."""
import json
import struct


def stamp(raw,licences):
    if not 9 <= len(raw) <= 64 << 20:
        raise ValueError("checkpoint byte ceiling")
    size = struct.unpack("<Q",raw[:8])[0]
    if not 0 < size <= 1 << 20 or 8+size > len(raw):
        raise ValueError("checkpoint header ceiling")
    header = json.loads(raw[8:8+size])
    before = {k:v for k,v in header.items() if k != "__metadata__"}
    metadata = header.setdefault("__metadata__",{})
    for key,value in licences.items():
        metadata["x-terms-"+key] = value
    new = json.dumps(header,separators=(",",":")).encode()
    new += b" "*(-len(new)%8)
    output = struct.pack("<Q",len(new))+new+raw[8+size:]
    if before != {k:v for k,v in json.loads(new).items() if k != "__metadata__"} or output[8+len(new):] != raw[8+size:]:
        raise ValueError("stamp changed tensor bytes")
    return output
