# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Primary-path static parser smoke without Foundation loading or neural imports."""
import argparse
import os
import subprocess
from pathlib import Path

from build import build
from contracts import FROZEN, HERE, SPEC, verify_recovery, write_json
from remote import stage_check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--go", required=True)
    args = parser.parse_args()
    root, kit = args.root.resolve(), args.kit.resolve()
    root.mkdir(parents=True, exist_ok=True)
    checked = subprocess.run(["git", "diff", "--exit-code", FROZEN["kit_commit"], "--", "pkg/adapterbuild",
                              "pkg/scoring/catalog", "pkg/wasm/runtime", "go.mod", "go.sum"], cwd=kit, check=False, timeout=30)
    if checked.returncode:
        raise ValueError("pinned parser source changed")
    environment = {**os.environ, "GOPROXY": "off", "GOSUMDB": "off", "GOTOOLCHAIN": "local", "CGO_ENABLED": "0",
                   "GOCACHE": str(root / "go-cache"), "GOTMPDIR": str(root), "TMPDIR": str(root)}
    subprocess.run([args.go, "build", "-mod=readonly", "-trimpath", "-o", str(root / "stagecheck"),
                    str(HERE / "stagecheck.go")], cwd=kit, env=environment, check=True, timeout=180)
    subprocess.run([args.go, "test", "-mod=readonly", "-count=1", "-v", str(HERE / "stagecheck.go"),
                    str(HERE / "stagecheck_test.go")], cwd=kit, env=environment, check=True, timeout=180)
    preservation = build(root)
    write_json(root / "foundation.json", SPEC)
    before = stage_check(root, root / "original.safetensors", FROZEN["origin_sha256"], "original-parser")
    after = stage_check(root, root / "SENTIMENT_ANALYSIS-recovery.safetensors", FROZEN["recovered_sha256"], "recovered-parser")
    payload = verify_recovery((root / "original.safetensors").read_bytes(), (root / "SENTIMENT_ANALYSIS-recovery.safetensors").read_bytes())
    proof = {"schema_version": 1, "origin_sha256": before["sha256"], "package_sha256": after["sha256"],
             "package_keccak": after["keccak256"].removeprefix("0x"), "package_bytes": after["bytes"],
             "tensor_payload_sha256": payload, "both_full_parsers_passed": True, "model_loaded": False,
             "source_snapshot_sha256": after["class_snapshot_sha256"], "scope": "declared Foundation layout only, no graph load",
             "byte_preservation": preservation}
    write_json(root / "static-proof.json", proof)
    print("BOTH_FULL_STATIC_PARSERS_PASS " + proof["package_keccak"])


if __name__ == "__main__":
    main()
