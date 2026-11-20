# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Exact owned serving inputs and restricted lease rows. No neural imports."""
import re

from lease_common import DEFAULT_OWNER, immutable_url
from ready_contract import validate_ready

FAMILY = "KEYWORD_EXTRACTION"
PREFIX = "https://raw.githubusercontent.com/zkasuran/telegraph-catalog-miners/"


def owned_url(value):
    immutable_url(value)
    remainder = value[len(PREFIX):]
    _, _, path = remainder.partition("/")
    if not path.startswith("verification/owners/keyword_extraction/"):
        raise ValueError("serving input outside the exact owned public namespace")
    return value


def serving_row(settings, report, commit):
    validate_ready(report)
    if not re.fullmatch(r"[a-f0-9]{64}", report["runtime"].get("executor_sha256", "")):
        raise ValueError("actual audited executor byte identity required")
    required = {"package_url", "report_url", "report_sha256", "slot", "forward_port", "runnable_url"}
    if set(settings) != required:
        raise ValueError("serving settings fields mismatch")
    if not re.fullmatch(r"[a-f0-9]{64}", settings["report_sha256"]):
        raise ValueError("exact report byte hash required")
    owned_url(settings["package_url"])
    owned_url(settings["report_url"])
    if report["serving_commit"] != commit:
        raise ValueError("serving must run the audited source commit")
    slot, port = settings["slot"], settings["forward_port"]
    if type(slot) is not int or not 1 <= slot <= 6 or type(port) is not int or port != 48600 + slot:
        raise ValueError("existing free slot and forward-port contract required")
    expected_url = "https://tga1.zkasuran.dev/_s2o_keyword_extraction_" + report["package_keccak"][:8] + "/runnable"
    if settings["runnable_url"] != expected_url:
        raise ValueError("exact installed adapter reservation route required")
    if report.get("owner", DEFAULT_OWNER).lower() != DEFAULT_OWNER:
        raise ValueError("selected Telegraph owner mismatch")
    for vector in report["runtime"]["vectors"]:
        if len(vector["tokens"]) > 384:
            raise ValueError("reference vector exceeds actual package token cap")
    proof = {"url": settings["report_url"], "sha256": settings["report_sha256"]}
    return {"family": FAMILY, "class": "judgment", "child": FAMILY + "_GENERAL", "owner": DEFAULT_OWNER,
            "file": FAMILY + ".safetensors", "url": settings["package_url"], "sha256": report["package_sha256"],
            "keccak": report["package_keccak"], "slot": slot, "forward_port": port, "runnable_url": expected_url,
            "verified_run_id": str(report["run_id"]), "verified_commit": report["commit"], "existing_registration_id": 0,
            "candidate_proof": dict(proof), "runtime_proof": dict(proof), "source_proof": dict(proof)}


def check_run(metadata, row):
    if str(metadata.get("id")) != row["verified_run_id"] or metadata.get("head_sha") != row["verified_commit"]:
        raise ValueError("actual audit run/commit mismatch")
    if metadata.get("status") != "completed" or metadata.get("conclusion") != "success" or metadata.get("event") != "workflow_dispatch":
        raise ValueError("completed passing audit workflow required")
    if metadata.get("path") != ".github/workflows/evaluator-owner-keyword_extraction.yml" or metadata.get("repository", {}).get("full_name") != "zkasuran/telegraph-catalog-miners" or metadata.get("repository", {}).get("private") is not False:
        raise ValueError("actual owned public audit workflow required")
