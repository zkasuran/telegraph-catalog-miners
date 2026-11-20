# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Immutable candidate evidence and original embedded grants, no model loading."""
import hashlib
import json
import re
from pathlib import Path

from data import HERE, digest
from licensing import APACHE_SHA, CC_SHA, MIT_SHA, OWN_SHA, bounded_raw, tensor_parts
from report import completed_checks

FREEZE_SHA = "c883f048ef5235da46cae54c1555c178bc8d927df63c2aff47aea2e3576276a1"
PACKAGE_SHA = "8e3ef949fea2fae5cfbe24f1e80ffd85d2bb77d947a2e610d4d01da062816814"
PACKAGE_KECCAK = "6ab216be4394b4c5208f8667d3577a64e6a3be6e275933d64d121b87a17e653b"


def read_frozen():
    raw = (HERE / "frozen-candidate.json").read_bytes()
    if digest(raw) != FREEZE_SHA:
        raise ValueError("original candidate freeze identity changed")
    value = json.loads(raw)
    if (value["training_disabled"] is not True or value["training_run_id"] != 44796102138
            or value["training_commit"] != "e5d4eaedf0cba90d2c837be94171d89dc29c8934"
            or value["package_sha256"] != PACKAGE_SHA or value["package_keccak"] != PACKAGE_KECCAK
            or value["audit_scored_once"] is not True or value["audit_used_for_selection"] is not False):
        raise ValueError("immutable original candidate required")
    for file in value["files"]:
        name = file["path"]
        if not re.fullmatch(r"frozen/[A-Za-z0-9_./-]+", name) or ".." in Path(name).parts:
            raise ValueError("frozen evidence path escape")
        path = HERE / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != file["bytes"] or digest(path.read_bytes()) != file["sha256"]:
            raise ValueError("frozen package or original report bytes changed")
    return value


def verify_frozen_package(path):
    freeze = read_frozen()
    raw = bounded_raw(path)
    if digest(raw) != freeze["package_sha256"] or len(raw) != freeze["package_bytes"]:
        raise ValueError("candidate package must remain byte-identical")
    header, _ = tensor_parts(raw)
    actual = header.get("__metadata__", {})
    wanted = json.loads((HERE / "frozen/licence-metadata.json").read_text())
    for key, pin in (("x-own-sand", OWN_SHA), ("x-upstream-apache-2.0", APACHE_SHA), ("x-upstream-mit", MIT_SHA), ("x-source-cc-by-4.0", CC_SHA)):
        if hashlib.sha256(wanted[key].encode()).hexdigest() != pin:
            raise ValueError("frozen original grant differs from reviewed terms")
    if any(actual.get(k) != v for k, v in wanted.items()):
        raise ValueError("candidate lacks the bound original complete grants and attribution")
    return freeze


def original_evidence():
    freeze = read_frozen()
    report = json.loads((HERE / "frozen/result.json").read_text())
    primary = json.loads((HERE / "frozen/data/primary-checks.json").read_text())
    completed_checks(report, primary)
    metadata = json.loads((HERE / "frozen/original-run.json").read_text())
    if (metadata["id"] != freeze["training_run_id"] or metadata["head_sha"] != freeze["training_commit"]
            or metadata["status"] != "completed" or metadata["conclusion"] != "failure"
            or metadata["event"] != "workflow_dispatch"):
        raise ValueError("original completed job identity mismatch")
    return freeze, report, primary


def source_check():
    freeze, report, primary = original_evidence()
    verify_frozen_package(HERE / "frozen/TEXT_SUMMARIZATION.safetensors")
    return {"frozen_candidate_source": "PASS", "training_run_id": freeze["training_run_id"],
            "package_sha256": freeze["package_sha256"], "training_disabled": True,
            "audit_rescored": False, "audit_groups": report["data"]["boundary"]["audit_groups"],
            "original_quality": {s: m["pairwise_accuracy"] for s, m in report["runtime"]["metrics"].items()}}


if __name__ == "__main__":
    print(json.dumps(source_check()))
