# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded inert-header rights verification. No tensor library or model loading."""
import json
import struct

from contracts import EXPERIMENT, HERE, sha


def check(raw, licences):
    if not 9 <= len(raw) <= 64 << 20:
        raise ValueError("checkpoint byte ceiling")
    length = struct.unpack("<Q", raw[:8])[0]
    if not 0 < length <= 1 << 20 or length + 8 > len(raw):
        raise ValueError("checkpoint inert header ceiling")
    header = json.loads(raw[8:8+length])
    metadata = header.get("__metadata__", {})
    for key in ("notice", "apache_2_0", "kit_mit", "sand"):
        if metadata.get("x-terms-" + key) != licences[key]:
            raise ValueError("checkpoint original terms changed: " + key)
    return True


def main():
    licences = json.loads((HERE / "support/training/licences.json").read_bytes())
    warm = EXPERIMENT["warm_start"]
    for name, expected in ((warm["checkpoint"], warm["checkpoint_sha256"]), (warm["untrained"], warm["untrained_sha256"])):
        raw = (HERE / "assets" / name).read_bytes()
        if sha(raw) != expected:
            raise ValueError("checkpoint source bytes changed")
        check(raw, licences)
    print("CHECKPOINT RIGHTS PASS: two immutable fitted/untrained containers, all original grants retained, no model loading")


if __name__ == "__main__":
    main()
