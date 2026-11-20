# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Preserve a clean audit until complete validation gains actually pass."""
import json

from data import HERE, VIEWS
from metrics import all_formats


def development_gate(trained, untrained, controls):
    if any(set(value) != set(VIEWS) for value in (trained, untrained, controls)):
        return {"passed": False, "audit_allowed": False, "reason": "missing development response format"}
    if any(value.get("n_eval") != 64 or value.get("independent_groups") != 32 for values in (trained, untrained) for value in values.values()):
        return {"passed": False, "audit_allowed": False, "reason": "complete independent development groups required"}
    quality = all_formats(trained, untrained, controls)
    return {"passed": quality["passed"], "audit_allowed": quality["passed"], "quality": quality,
            "audit_scored": False, "selection_split": "validation"}


def check_source_quarantine():
    exposure = json.loads((HERE / "exposed-audit.json").read_text())
    contract = json.loads((HERE / "contract.json").read_text())
    manifest = json.loads((HERE / "data/manifest.json").read_text())
    if contract["splits"]["audit"]["sha256"] == exposure["audit_sha256"]:
        raise ValueError("previous scored audit freeze reused")
    documents = {d["doi"] for groups in manifest["splits"].values() for g in groups for d in g["documents"]}
    entities = {e for groups in manifest["splits"].values() for g in groups for e in g["source_entity_ids"]}
    if documents & set(exposure["dois"]) or entities & set(exposure["source_entity_ids"]):
        raise ValueError("previous scored audit source/entity overlap")
    return {"source_quarantine": "PASS", "source_entity_overlap": 0, "audit_scored": False}


if __name__ == "__main__":
    print(json.dumps(check_source_quarantine()))
