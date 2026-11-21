# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bound metadata and public lease bytes before preparing adapter activation."""
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime

FAMILY = "CONTENT_MODERATION"
WORKFLOW = ".github/workflows/evaluator-owner-content_moderation.yml"
MAX_ZIP = 65536
MAX_LEASE = 8192


def continued_dispatch(receipt, envelope, original_dispatch_key):
    request = envelope.get("request", {})
    if (receipt.get("family") != FAMILY or request.get("family") != FAMILY or request.get("action") != "reconcile"
            or request.get("payload") != {"operation_key": original_dispatch_key}
            or receipt.get("key") != envelope.get("key")
            or receipt.get("request_sequence") != envelope.get("submitted_at")):
        raise ValueError("continuation receipt/request literal identity differs")
    if receipt.get("status") == "waiting_capacity":
        if receipt.get("result", {}).get("status") != "capacity_wait" or receipt["result"].get("retry_allowed") is not True:
            raise ValueError("capacity wait lacks actual continuation permission")
        return None
    if receipt.get("status") != "succeeded" or not isinstance(receipt.get("result"), dict):
        raise ValueError("unknown/failed continuation is not permission to redispatch")
    return receipt["result"]


def select_run(rows, payload, started_at):
    if type(started_at) not in (int, float) or started_at <= 0 or payload.get("workflow") != WORKFLOW:
        raise ValueError("actual dispatch time/workflow boundary required")
    matches = []
    for row in rows:
        if (row.get("event") == "workflow_dispatch" and row.get("path") == WORKFLOW
                and row.get("head_sha") == payload["commit"]):
            moment = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            if moment.tzinfo is None:
                raise ValueError("remote run timestamp lacks timezone")
            if moment.timestamp() >= started_at - 2:
                if type(row.get("id")) is not int or row["id"] <= 0:
                    raise ValueError("actual integer GitHub run required")
                matches.append(row)
    if len(matches) != 1:
        raise ValueError("dispatch run must be unique across the complete relevant listing")
    return matches[0]


def select_artifact(rows, run_id):
    expected = "owner-lease-content_moderation-" + str(run_id)
    matches = [row for row in rows if row.get("name") == expected]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("duplicate current public lease artifacts")
    artifact = matches[0]
    if (artifact.get("expired") is not False or artifact.get("workflow_run", {}).get("id") != run_id
            or type(artifact.get("id")) is not int or artifact["id"] <= 0
            or type(artifact.get("size_in_bytes")) is not int or not 0 < artifact["size_in_bytes"] <= MAX_ZIP
            or not re.fullmatch(r"sha256:[a-f0-9]{64}", artifact.get("digest", ""))):
        raise ValueError("actual lease artifact identity/digest/size differs")
    return artifact


def lease_bytes(raw, artifact):
    if len(raw) != artifact["size_in_bytes"] or len(raw) > MAX_ZIP or "sha256:" + hashlib.sha256(raw).hexdigest() != artifact["digest"]:
        raise ValueError("complete public lease ZIP bytes differ")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if (len(entries) != 1 or entries[0].filename != "lease.json" or entries[0].flag_bits & 1
                    or not 0 < entries[0].file_size <= MAX_LEASE):
                raise ValueError("exact one-file restricted public lease required")
            data = archive.read(entries[0])
    except (zipfile.BadZipFile, RuntimeError, OSError) as error:
        raise ValueError("invalid bounded public lease archive") from error
    if len(data) > MAX_LEASE:
        raise ValueError("public lease JSON exceeds ceiling")
    def unique(fields):
        value = {}
        for key, item in fields:
            if key in value:
                raise ValueError("duplicate public lease JSON field")
            value[key] = item
        return value
    lease = json.loads(data, object_pairs_hook=unique)
    if not isinstance(lease, dict):
        raise ValueError("public lease must be an object")
    return lease, data


def activation_payload(lease, run, artifact, candidate, validate_lease, row, clock=None):
    if (run.get("status") != "in_progress" or run.get("head_sha") != candidate["serving_commit"]
            or type(run.get("id")) is not int or artifact.get("workflow_run", {}).get("id") != run["id"]
            or lease.get("run_id") != str(run["id"]) or lease.get("commit") != run["head_sha"]):
        raise ValueError("actual public lease and active source/run differ")
    # Use the real common.lease_for/validate_lease restriction and expiry contract.
    validate_lease(lease, row, clock)
    return {"package_keccak": candidate["package_keccak"], "package_sha256": candidate["package_sha256"],
            "serving_run_id": run["id"], "serving_commit": run["head_sha"], "lease_artifact_id": artifact["id"],
            "candidate_report_sha256": candidate["report_sha256"]}
