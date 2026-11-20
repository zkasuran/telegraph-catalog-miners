# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Byte-preserving, source-only recovery export. No model loading or scoring."""
import argparse
from pathlib import Path

from contracts import (
    FROZEN,
    HERE,
    fetch,
    recover_package,
    sha,
    strict_json,
    verify_recovery,
    write_json,
)


def rights():
    path = HERE / "licences.json"
    fetch(FROZEN["licences_url"], path, FROZEN["licences_sha256"], 1 << 20)
    value = strict_json(path.read_bytes())
    if "LicenseRef-zkasuran-SAND-1.0" not in value["sand"] or "Permission is hereby granted" not in value["kit_mit"]:
        raise ValueError("reviewed outbound/upstream grant missing")
    (HERE / "LICENSE").write_text(value["sand"])
    (HERE / "NOTICE").write_text(value["notice"] + "\n\nRegistration recovery preserves all upstream grants and every original tensor byte.\n")


def build(root):
    root.mkdir(parents=True, exist_ok=True)
    source = root / "original.safetensors"
    fetch(FROZEN["origin_url"], source, FROZEN["origin_sha256"])
    raw = source.read_bytes()
    recovered = recover_package(raw, FROZEN["recovery_id"])
    target = root / "SENTIMENT_ANALYSIS-recovery.safetensors"
    target.write_bytes(recovered)
    proof = {"schema_version": 1, "family": FROZEN["family"], "recovery_id": FROZEN["recovery_id"],
             "origin_sha256": sha(raw), "package_sha256": sha(recovered), "package_bytes": len(recovered),
             "tensor_payload_sha256": verify_recovery(raw, recovered), "all_original_metadata_retained": True,
             "all_tensor_descriptors_retained": True, "all_tensor_bytes_retained": True,
             "upstream_grants_retained": True, "model_loaded": False,
             "lineage": "own trained delta; ignored registration metadata only, not a foreign binary wrap",
             "original_training_run": FROZEN["training_run_id"], "original_runtime_run": FROZEN["runtime_run_id"],
             "runtime_reproduction": "pending bounded public CPU workflow"}
    write_json(root / "byte-preservation.json", proof)
    return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rights", action="store_true")
    args = parser.parse_args()
    if args.rights:
        rights()
    proof = build(args.out)
    print("SOURCE_ONLY_RECOVERY_EXPORTED " + proof["package_sha256"])


if __name__ == "__main__":
    main()
