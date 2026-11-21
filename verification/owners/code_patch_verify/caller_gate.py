# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Materialised caller-proof bindings, not deployment authentication.

Affirmative flags and hash-shaped strings are not evidence. This gate requires
the actual source, grant, literal trace and CI observation bytes. An independent
deployed-caller proof is still required; synthetic test material is not one.
"""
import hashlib
import re
from pathlib import Path

from ci_reference import EvidenceError, expected, strict_json

MAX_PROOF_BYTES = 20000
MAX_MATERIAL_BYTES = 1 << 20
ROLES = {"caller_source": "source", "caller_licence": "grants", "literal_trace": "trace", "ci_observation": "observations"}
MATERIAL_PATHS = {"caller_source": "caller-proof/source/caller.go", "caller_licence": "caller-proof/grants/LICENSE.txt",
                  "literal_trace": "caller-proof/trace/capture.json", "ci_observation": "caller-proof/observations/ci.json"}


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _same_json_value(left, right):
    """Keep native JSON types distinct without normalizing literal strings."""
    pending = [(left, right)]
    while pending:
        left, right = pending.pop()
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            if left.keys() != right.keys():
                return False
            pending.extend((value, right[key]) for key, value in left.items())
        elif isinstance(left, list):
            if len(left) != len(right):
                return False
            pending.extend(zip(left, right))
        elif left != right:
            return False
    return True


def _material(root, role, record):
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("caller material needs an exact path/hash binding")
    if not isinstance(record["path"], str) or not isinstance(record["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", record["sha256"]):
        raise ValueError("invalid caller material binding")
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 3 or relative.parts[:2] != ("caller-proof", ROLES[role]):
        raise ValueError("caller material must stay in its declared proof role directory")
    forbidden = {"ACCOUNTS.md", "auth.json", "id_rsa", "id_ed25519"}
    if any(part.startswith(".") or part in forbidden for part in relative.parts):
        raise ValueError("protected/private source is not caller evidence")
    if record["path"] != MATERIAL_PATHS[role]:
        raise ValueError("only the fixed declared caller material files may be read")
    for index in range(1, len(relative.parts) + 1):
        if (root / Path(*relative.parts[:index])).is_symlink():
            raise ValueError("caller proof symlink refused")
    path = root / relative
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_MATERIAL_BYTES:
        raise ValueError("missing or oversized caller material")
    raw = path.read_bytes()
    if _hash(raw) != record["sha256"]:
        raise ValueError("caller material hash mismatch: " + role)
    return raw


def validate_caller_contract(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_PROOF_BYTES:
        raise ValueError("missing bounded independently verified caller contract")
    record = strict_json(path.read_text())
    if not isinstance(record, dict) or type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ValueError("caller contract schema must be an object with version 1")
    for key, size in (("caller_source_sha256", 64), ("caller_revision", 40), ("sample_sha256", 64)):
        if not isinstance(record.get(key), str) or not re.fullmatch(r"[a-f0-9]{" + str(size) + "}", record[key]):
            raise ValueError("missing caller source/sample identity")
    if (record.get("family") != "CODE_PATCH_VERIFY" or record.get("class") != "judgment"
            or record.get("trusted_ci_observation_in_literal_prompt") is not True or record.get("protected_rows_used") is not False
            or record.get("sample_origin") != "independently_constructed_canonical_request"):
        raise ValueError("caller origin/trust claims are incomplete")
    prompt = record.get("literal_sample_prompt")
    if not isinstance(prompt, str) or _hash(prompt.encode()) != record["sample_sha256"]:
        raise ValueError("literal caller prompt hash mismatch")
    bindings = record.get("evidence_files")
    if not isinstance(bindings, dict) or set(bindings) != set(ROLES):
        raise ValueError("flags are not evidence: source, grant, trace and observation files required")
    raw = {role: _material(path.parent, role, bindings[role]) for role in ROLES}
    if _hash(raw["caller_source"]) != record["caller_source_sha256"]:
        raise ValueError("caller source identity differs from materialised source")
    trace = strict_json(raw["literal_trace"].decode())
    observation = strict_json(raw["ci_observation"].decode())
    sample = strict_json(prompt)
    if not isinstance(trace, dict) or not isinstance(sample, dict):
        raise ValueError("caller trace/sample must be objects")
    if trace.get("protected_rows_used") is not False:
        raise ValueError("trace protected-row boundary must be a native false boolean")
    identities = {"caller_revision": record["caller_revision"], "caller_source_sha256": record["caller_source_sha256"],
                  "source_observation_sha256": _hash(raw["ci_observation"]), "prompt": prompt,
                  "sample_origin": record["sample_origin"], "protected_rows_used": False}
    if any(trace.get(key) != value for key, value in identities.items()):
        raise ValueError("captured literal trace does not bind caller/source/sample identity")
    if not _same_json_value(sample.get("ci_observation"), observation) or not isinstance(sample.get("patch"), str):
        raise ValueError("literal sample does not preserve the bound CI observation")
    try:
        expected({"patch": sample["patch"]}, observation)
    except EvidenceError as exc:
        raise ValueError("bound CI observation identity/state is invalid") from exc
    # Necessary material checks do not turn an invented trace into a real one.
    return {**record, "byte_bound_material_roles": sorted(raw),
            "deployment_authentication": "not_established_by_this_validator"}
