# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only contract check of the exact observed public readiness proof."""
import json

from data import HERE, digest
from frozen_candidate import original_evidence
from report import validate_ready

MANIFEST_SHA = "7b4746deb324f8651ae16a8f30316cef5e7d74737ee7d474cfdf37c9ab498671"
READY_SHA = "5a92cb05d9dcf00ddb5eaafe19234ee2f66234433400f077011b34b5e70defa9"


def source_check():
    raw = (HERE / "release-manifest.json").read_bytes()
    if digest(raw) != MANIFEST_SHA:
        raise ValueError("observed release manifest identity changed")
    release = json.loads(raw)
    for item in release["files"]:
        path = HERE / item["path"]
        if path.is_symlink() or path.stat().st_size != item["bytes"] or digest(path.read_bytes()) != item["sha256"]:
            raise ValueError("observed public proof bytes changed")
    report_raw = (HERE / "release/registration-ready.json").read_bytes()
    if digest(report_raw) != READY_SHA:
        raise ValueError("actual ready report identity changed")
    report = validate_ready(json.loads(report_raw))
    freeze, original, _ = original_evidence()
    recovery = json.loads((HERE / "release/recovery-result.json").read_text())
    if (report["run_id"] != release["verified_run_id"] or report["commit"] != release["verified_commit"]
            or report["serving_commit"] != release["serving_commit"] or release["ready_artifact_id"] != 14675422884
            or report["package_sha256"] != freeze["package_sha256"] or report["package_keccak"] != freeze["package_keccak"]
            or recovery["status"] != "PASS" or recovery["fitting_performed"] is not False or recovery["audit_rescored"] is not False
            or report["training_origin"] != recovery["original_training"]
            or report["runtime"]["vectors"] != original["runtime"]["vectors"]):
        raise ValueError("observed original/recovery/serving proof bindings changed")
    return {"released_ready_source": "PASS", "verified_run_id": report["run_id"], "verified_commit": report["commit"],
            "serving_commit": report["serving_commit"], "package_sha256": report["package_sha256"],
            "ready_sha256": READY_SHA, "canonical_registration": "not attempted"}


if __name__ == "__main__":
    print(json.dumps(source_check()))
