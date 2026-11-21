# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Exact next-lease source artifacts. No runtime, signing or model operations."""
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = "CONTENT_MODERATION"
REPO = "zkasuran/telegraph-catalog-miners"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def validate(value, report_raw, lease_raw):
    required = {"schema_version", "license", "family", "generation", "previous_serving_run_id", "previous_lease_expires_at",
                "previous_lease_sha256", "package_sha256", "package_keccak", "report_sha256", "audit_run_id", "audit_commit",
                "serving_commit", "report_path", "package_url", "source_publication_key", "scope"}
    if not isinstance(value, dict) or set(value) != required or value["schema_version"] != 1 or value["family"] != FAMILY:
        raise ValueError("lease continuation descriptor schema/family differs")
    if value["license"] != "LicenseRef-zkasuran-SAND-1.0" or value["generation"] != "lease_" + str(value["previous_serving_run_id"]):
        raise ValueError("continuation requires a licensed actual prior-run generation")
    if type(value["previous_serving_run_id"]) is not int or value["previous_serving_run_id"] < 1:
        raise ValueError("actual previous serving run required")
    report, lease = json.loads(report_raw), json.loads(lease_raw)
    if sha(report_raw) != value["report_sha256"] or sha(lease_raw) != value["previous_lease_sha256"]:
        raise ValueError("original report or public lease bytes changed")
    if report.get("status") != "REGISTRATION_READY" or report.get("family") != FAMILY:
        raise ValueError("original passing audit required; continuation cannot claim a new audit")
    if report.get("run_id") != value["audit_run_id"] or report.get("commit") != value["audit_commit"]:
        raise ValueError("original audit identity differs")
    for name in ("package_sha256", "package_keccak", "serving_commit"):
        if value[name] != report.get(name):
            raise ValueError("continuation changes qualified package or executor source")
    if lease.get("family") != FAMILY or str(value["previous_serving_run_id"]) != lease.get("run_id") or lease.get("commit") != value["serving_commit"]:
        raise ValueError("prior lease belongs to another family, runner or source")
    if lease.get("adapter_hash") != value["package_keccak"] or lease.get("package_sha256") != value["package_sha256"]:
        raise ValueError("prior lease package differs")
    if lease.get("expires_at") != value["previous_lease_expires_at"] or datetime.fromisoformat(value["previous_lease_expires_at"]).tzinfo is None:
        raise ValueError("actual explicit lease expiry required")
    if any(lease.get(kind + "_proof_sha256") != value["report_sha256"] for kind in ("candidate", "runtime", "source")):
        raise ValueError("lease source/runtime/candidate proof identity differs")
    if value["report_path"] != "verification/owners/content_moderation/continuations/" + value["generation"] + "/registration-ready.json":
        raise ValueError("report is not the explicit owned next-lease artifact")
    if not re.fullmatch(r"[a-f0-9]{64}", value["source_publication_key"]):
        raise ValueError("actual original serving publication identity required")
    if not re.fullmatch(r"https://raw\.githubusercontent\.com/" + re.escape(REPO) + r"/[a-f0-9]{40}/verification/owners/content_moderation/candidate/CONTENT_MODERATION\.safetensors", value["package_url"]):
        raise ValueError("package must remain its original immutable public bytes")
    return lease


def dispatch_payload(value, report_raw, lease_raw, continuation_commit):
    lease = validate(value, report_raw, lease_raw)
    if not re.fullmatch(r"[a-f0-9]{40}", continuation_commit):
        raise ValueError("confirmed immutable continuation publication required")
    inputs = {"mode": "serve", "campaign_exclusions_json": "[]", "package_url": value["package_url"],
              "package_sha256": value["package_sha256"], "package_keccak": value["package_keccak"],
              "report_url": "https://raw.githubusercontent.com/" + REPO + "/" + continuation_commit + "/" + value["report_path"],
              "report_sha256": value["report_sha256"], "slot": str(lease["slot"]), "forward_port": str(lease["forward_port"]),
              "runnable_url": lease["runnable_url"]}
    return {"workflow": ".github/workflows/evaluator-owner-content_moderation.yml", "commit": value["serving_commit"],
            "inputs": inputs, "publication_key": value["source_publication_key"]}


def main():
    directory = HERE / "continuations/lease_44827170108"
    value = json.loads((directory / "continuation.json").read_bytes())
    report_raw, lease_raw = (directory / "registration-ready.json").read_bytes(), (directory / "previous-lease.json").read_bytes()
    validate(value, report_raw, lease_raw)
    if report_raw != (HERE / "candidate/registration-ready.json").read_bytes():
        raise ValueError("lease generation changed the original audit report")
    print("LEASE CONTINUATION SOURCE PASS: original audit/package/executor and public prior lease bound; no new job, audit or activation claimed")


if __name__ == "__main__":
    main()
