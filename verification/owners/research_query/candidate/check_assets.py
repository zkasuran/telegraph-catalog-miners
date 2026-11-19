# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Static exact-byte checks for one remotely audited released delta."""
import hashlib
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from report import validate_ready


def verify_assets(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    report_raw = (directory / "registration-ready.json").read_bytes()
    report = json.loads(report_raw)
    validate_ready(report)
    package = (directory / "RESEARCH_QUERY.safetensors").read_bytes()
    if len(package) > 64 << 20 or hashlib.sha256(package).hexdigest() != manifest["package_sha256"] or len(package) != manifest["package_bytes"]:
        raise ValueError("released delta differs from audited bytes")
    if hashlib.sha256(report_raw).hexdigest() != manifest["report_sha256"] or report["package_sha256"] != manifest["package_sha256"] or report["package_keccak"] != manifest["package_keccak"]:
        raise ValueError("released report or package identity changed")
    if report["run_id"] != manifest["verified_run_id"] or report["commit"] != manifest["verified_commit"]:
        raise ValueError("released assets do not bind the actual successful run")
    size = struct.unpack("<Q", package[:8])[0]
    if not 0 < size <= 1 << 20 or 8 + size >= len(package):
        raise ValueError("bounded package metadata required")
    metadata = json.loads(package[8:8 + size])["__metadata__"]
    if hashlib.sha256(metadata["x-own-sand"].encode()).hexdigest() != "e6cf0cc83fbc51616590209d71ef1fa5fe46c130bef44e0d97d439ee91663447":
        raise ValueError("actual package lost its reviewed own terms")
    attribution = json.loads(metadata["x-source-attribution"])
    if len(attribution) != 112 or any(not row["license_uri"] or not row["authors"] for row in attribution):
        raise ValueError("original source grants or attribution missing")
    return {"package_sha256": manifest["package_sha256"], "package_keccak": manifest["package_keccak"],
            "report_sha256": manifest["report_sha256"], "verified_run_id": manifest["verified_run_id"],
            "verified_commit": manifest["verified_commit"], "static_bytes_passed": True,
            "neural_operations": 0, "canonical_claim": False}


if __name__ == "__main__":
    print(json.dumps(verify_assets(HERE)))
