# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Combine only observed passing engine audits into a registration-ready report."""
import argparse
import os
import re
from pathlib import Path

from assets import FAMILY, HERE, SCOPE, remote_only
from common import DEFAULT_OWNER, FOUNDATION, KIT, public_json, sha, strict_json, write_json
from contracts import FORMATS, validate_quality_runtime, validate_ready
from verify import source_proof
from parent_scope import claimed_intents, material_contract, verify_scope


def verified_package_bytes(directory, candidate):
    package = candidate.get("package") if isinstance(candidate, dict) else None
    if not isinstance(package, dict) or package.get("file") != FAMILY + ".safetensors":
        raise ValueError("exact parent-owned package filename required")
    expected, size = package.get("sha256"), package.get("bytes")
    if (not isinstance(expected, str) or re.fullmatch(r"[a-f0-9]{64}", expected) is None or
            type(size) is not int or not 1 <= size <= 64 << 20):
        raise ValueError("bounded package identity/count required")
    path = directory / package["file"]
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
        raise ValueError("result package file/type/count changed")
    with path.open("rb") as source:
        raw = source.read((64 << 20) + 1)
    if len(raw) != size or sha(raw) != expected:
        raise ValueError("result package bytes changed")
    return raw


def emit_verified_result(candidate, reports, directory, output):
    # Verify once before ready emission and copy the checked buffer, not a later
    # file read. This helper never loads a model; tests use source-only bytes.
    raw = verified_package_bytes(directory, candidate)
    (output.parent / candidate["package"]["file"]).write_bytes(raw)
    ready = combine(candidate, reports, output)
    return ready


def combine(candidate, reports, output):
    verify_scope(candidate, SCOPE)
    source_hash = sha((HERE / "frozen/frozen.json").read_bytes())
    if candidate.get("family") != FAMILY or candidate.get("source_manifest_sha256") != source_hash or candidate.get("audit_read") is not False:
        raise ValueError("frozen candidate/source boundary mismatch")
    if set(reports) != set(FORMATS):
        raise ValueError("all claimed engine audit reports required")
    for name, report in reports.items():
        verify_scope(report, SCOPE)
        if (report.get("family") != FAMILY or report.get("format") != name or report.get("passed") is not True or
                report.get("run_id") != candidate["run_id"] or report.get("commit") != candidate["commit"] or
                report.get("package_sha256") != candidate["package"]["sha256"] or
                report.get("source_manifest_sha256") != source_hash or report.get("foundation") != FOUNDATION or
                report.get("kit_commit") != KIT or report.get("quality_engine") != "pinned reference WASM, no native Torch audit"):
            raise ValueError("measured engine report identity/quality mismatch")
    first = reports["json"]
    if any(r["package_keccak"] != first["package_keccak"] or r["runtime"]["vectors"] != first["runtime"]["vectors"] or
           r["executor_sha256"] != first["executor_sha256"] for r in reports.values()):
        raise ValueError("cross-run actual executor or repeated-vector identity mismatch")
    measured_source = source_proof()
    if any(r["independent_groups"] != measured_source["independent_groups"] or r["audit_pairs"] != measured_source["audit_pairs"] for r in reports.values()):
        raise ValueError("measured audit/source group identity mismatch")
    ready = {"schema_version": 1, "status": "REGISTRATION_READY", "family": FAMILY,
             "scope": SCOPE, "claimed_intents": claimed_intents(SCOPE),
             "run_id": candidate["run_id"], "commit": candidate["commit"], "serving_commit": candidate["commit"],
             "package_sha256": candidate["package"]["sha256"], "package_keccak": first["package_keccak"],
             "package_bytes": candidate["package"]["bytes"], "owner": DEFAULT_OWNER,
             "foundation_sha256": FOUNDATION["foundation_lora_fp32.onnx"],
             "engine_sha256": FOUNDATION["foundation_engine.wasm"], "tokenizer_sha256": FOUNDATION["tokenizer.json"],
             "source": {**measured_source,
                        "scope": "historical official award-list entailment with supplied evidence, not universal event prediction"},
             "quality": {"passed": True, "audit_pairs": min(r["audit_pairs"] for r in reports.values()),
                         "all_supported_formats_tested": True, "formats": {name: r["metrics"] for name, r in reports.items()}},
             "runtime": first["runtime"], "executor_sha256": first["executor_sha256"], "kit_commit": KIT,
             "runtime_versions": {name: r["runtime_versions"] for name, r in reports.items()},
             "class_snapshot_sha256": {name: r["class_snapshot_sha256"] for name, r in reports.items()},
             "audit_report_sha256": {name: sha(__import__("json").dumps(r, separators=(",", ":"), allow_nan=False).encode() + b"\n") for name, r in reports.items()},
              "SPDX-License-Identifier": "LicenseRef-zkasuran-SAND-1.0"}
    validate_quality_runtime(ready)
    if SCOPE == "parent_only":
        status, raw_status = public_json("https://devnode.telegraphprotocol.com/api/evalpool/status?intent_id=" + FAMILY)
        material = material_contract(status)
        ready["source"]["parent_material"] = material
        ready["source"]["parent_material_status_sha256"] = sha(raw_status)
        if not material["material_ready"]:
            write_json(output.parent / "material-condition.json", {
                "schema_version": 1, "status": "WAITING_PARENT_MATERIAL", "family": FAMILY,
                "scope": SCOPE, "run_id": candidate["run_id"], "commit": candidate["commit"],
                "package_sha256": candidate["package"]["sha256"], "material": material,
                "material_status_sha256": sha(raw_status), "registration_ready": False,
                "SPDX-License-Identifier": "LicenseRef-zkasuran-SAND-1.0"})
            raise ValueError("parent material readiness has not passed; candidate/audit remain preserved")
    validate_ready(ready)
    raw = __import__("json").dumps(ready, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    if len(raw) > 64 << 10:
        raise ValueError("ready report byte ceiling")
    output.write_bytes(raw)
    return ready


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--audits", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    candidate = strict_json((args.candidate_dir / "candidate.json").read_bytes())
    if candidate["run_id"] != int(os.environ["GITHUB_RUN_ID"]) or candidate["commit"] != os.environ["GITHUB_SHA"]:
        raise ValueError("actual GitHub run/commit mismatch")
    reports = {name: strict_json((args.audits / ("audit-" + name + ".json")).read_bytes()) for name in FORMATS}
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("LICENSE", "NOTICE", "DATA-SOURCES.md", "UPSTREAM-LICENCES.json"):
        (args.out / name).write_bytes((HERE / name).read_bytes())
    ready = emit_verified_result(candidate, reports, args.candidate_dir, args.out / "registration-ready.json")
    for name in ("LICENSE", "NOTICE", "DATA-SOURCES.md", "UPSTREAM-LICENCES.json"):
        (args.out / name).write_bytes((HERE / name).read_bytes())
    write_json(args.out / "result-bindings.json", {"run_id": ready["run_id"], "commit": ready["commit"],
               "candidate_sha256": sha((args.candidate_dir / "candidate.json").read_bytes()),
               "registration_ready_sha256": sha((args.out / "registration-ready.json").read_bytes())})
    print("REGISTRATION_READY from completed independent reference-engine audits")


if __name__ == "__main__":
    main()
