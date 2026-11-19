# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Generate local full licence sidecars from already verified primary terms."""
import argparse
import json
from pathlib import Path

from data import HERE, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-license", required=True, type=Path)
    parser.add_argument("--own-license", type=Path, default=HERE.parents[1] / "LICENSE")
    args = parser.parse_args()
    raw = args.source_license.read_bytes()
    pins = json.loads((HERE / "sources.json").read_text())
    if digest(raw) != pins["files"]["oasst-license.txt"][1]:
        raise ValueError("Apache source terms hash mismatch")
    own = args.own_license.read_bytes()
    if digest(own) != "e6cf0cc83fbc51616590209d71ef1fa5fe46c130bef44e0d97d439ee91663447":
        raise ValueError("reviewed catalog SAND terms hash mismatch")
    (HERE / "UPSTREAM-APACHE-2.0.txt").write_bytes(raw)
    (HERE / "LICENSE").write_bytes(own)
    shared = json.loads((HERE.parent / "training/licences.json").read_text())
    (HERE / "KIT-LICENSE").write_text(shared["kit_mit"])


if __name__ == "__main__":
    main()
