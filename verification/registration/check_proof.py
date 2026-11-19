# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only verification of retained public recovery proof, without runtime."""
import argparse
from pathlib import Path

from common import ANCHORS, MAX_BODY, check_proofs, sha, strict_json


def check(family, paths):
    anchor = ANCHORS[family]
    row = {**anchor, "family": family, "child": family + "_GENERAL"}
    values = []
    for name, path in zip(("candidate", "runtime", "source"), paths):
        if path.stat().st_size > MAX_BODY:
            raise ValueError("public proof exceeds its byte ceiling")
        raw = path.read_bytes()
        if sha(raw) != anchor[name + "_sha256"]:
            raise ValueError("retained verified proof hash mismatch")
        row[name + "_proof"] = {"sha256": anchor[name + "_sha256"]}
        values.append(strict_json(raw))
    check_proofs(row, *values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(ANCHORS), required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    check(args.family, [args.candidate, args.runtime, args.source])
    print("SOURCE_ONLY_RECOVERY_PROOF_PASS " + args.family)


if __name__ == "__main__":
    main()
