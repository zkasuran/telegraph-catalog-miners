# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only checks for the exact remotely qualified release. No model imports."""
import json
import struct

from contracts import FAMILY, HERE, sha, validate_report


def validate_bytes(package, report, parser, source, licences):
    vectors = validate_report(report)
    if report["package_sha256"] != sha(package) or report["package_bytes"] != len(package):
        raise ValueError("candidate package hash/length differs from actual report")
    if parser["sha256"] != sha(package) or parser["bytes"] != len(package) or parser["keccak256"] != "0x" + report["package_keccak"] or parser["manifest"]["verification"] != {"loaded": False, "stage1": True}:
        raise ValueError("candidate differs from actual full Go parser result")
    length = struct.unpack("<Q", package[:8])[0]
    if not 0 < length <= 1 << 20 or length + 8 >= len(package):
        raise ValueError("candidate package inert-header ceiling")
    meta = json.loads(package[8:8+length])["__metadata__"]
    for name, key in (("notice", "x-upstream-notice"), ("apache_2_0", "x-upstream-apache-2.0"), ("kit_mit", "x-upstream-mit"), ("sand", "x-licence")):
        if meta.get(key) != licences[name]:
            raise ValueError("candidate original grant changed")
    if sha(source) != report["source"]["source_manifest_sha256"]:
        raise ValueError("candidate source proof changed")
    return vectors


def main():
    root = HERE / "candidate"
    package = (root / (FAMILY + ".safetensors")).read_bytes()
    report = json.loads((root / "registration-ready.json").read_bytes())
    parser = json.loads((root / (FAMILY + ".safetensors.parser.json")).read_bytes())
    source = (root / "source-manifest.json").read_bytes()
    licences = json.loads((HERE / "support/training/licences.json").read_bytes())
    vectors = validate_bytes(package, report, parser, source, licences)
    if report["run_id"] != 44787260374 or report["commit"] != "d1c243ddf46bf3f3f939bcccbd67dc659ea6f577":
        raise ValueError("qualified candidate origin job changed")
    if sum(path.stat().st_size for path in root.iterdir() if path.is_file()) > 15 << 20:
        raise ValueError("bounded candidate release ceiling")
    print("CANDIDATE BYTES PASS: exact package/report/parser/source/grants, " + str(len(vectors)) + " real reference vectors, no local inference")


if __name__ == "__main__":
    main()
