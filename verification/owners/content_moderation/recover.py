# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Complete interrupted F3 qualification without fitting or changing its package."""
import argparse
import io
import json
import os
import re
import struct
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from baselines import scores as shortcut_scores
from contracts import EXPERIMENT, FAMILY, FORMATS, FOUNDATION, HERE, bits, eligibility, remote_only, sha, strict_json, write
from remote import capture_class, compile_tools, engine_audit, execute_phase, finish_report, read, support


def validate_plan(plan):
    if plan.get("schema_version") != 1 or plan.get("family") != FAMILY or plan.get("origin_run_id") != 44703787106 or plan.get("origin_commit") != "77bc762572b786f79b50e5652d1aeb3df19613e1":
        raise ValueError("exact frozen F3 origin required")
    if plan.get("artifact_id") != 14621277811 or plan.get("archive_sha256") != "af300054bcbf5659ab58caf80bb17720954d5769d75325770e60e687558eb117" or plan.get("archive_bytes") != 42738840:
        raise ValueError("original whole artifact identity changed")
    if plan.get("url") != "https://nightly.link/zkasuran/telegraph-catalog-miners/actions/artifacts/14621277811.zip" or plan.get("audit_sha256") != EXPERIMENT["audit_sha256"] or plan.get("phase_cap_seconds") != 12600:
        raise ValueError("recovery source, audit or CPU budget changed")
    if plan.get("trained_score_streams") != {name: 512 for name in FORMATS} or len(plan.get("files", [])) != 23:
        raise ValueError("complete original measurement inputs required")
    names = set()
    for row in plan["files"]:
        name = row.get("name", "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in names or row.get("member") != "cm-training/evidence/" + name or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")) or type(row.get("bytes")) is not int or not 0 < row["bytes"] <= 16 << 20:
            raise ValueError("invalid or duplicate recovery asset")
        names.add(name)


def archive_members(raw, plan):
    """Closed ZIP reader. Returns only hash-bound public source/model artifacts."""
    if len(raw) != plan["archive_bytes"] or sha(raw) != plan["archive_sha256"]:
        raise ValueError("whole origin archive hash/size mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if len(members) > 200 or sum(row.file_size for row in members) > 120 << 20:
            raise ValueError("origin archive count/expansion ceiling")
        seen = set()
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename or member.filename in seen or member.file_size > 16 << 20 or (member.external_attr >> 16 & 0o170000) == 0o120000:
                raise ValueError("unsafe or duplicate origin archive member")
            seen.add(member.filename)
        result = {}
        for row in plan["files"]:
            value = archive.read(row["member"])
            if len(value) != row["bytes"] or sha(value) != row["sha256"]:
                raise ValueError("bound origin member changed")
            result[row["name"]] = value
        return result


def verify_origin(files, plan):
    binding = strict_json(files["run-binding.json"])
    if binding["family"] != FAMILY or binding["run_id"] != plan["origin_run_id"] or binding["commit"] != plan["origin_commit"] or binding["experiment_sha256"] != plan["experiment_sha256"]:
        raise ValueError("original model job identity changed")
    frozen = strict_json(files["candidate-freeze.json"])
    if frozen["experiment_sha256"] != sha((HERE / "experiment.json").read_bytes()) or frozen["candidate"]["checkpoint_sha256"] != plan["candidate_checkpoint_sha256"] or frozen["source_manifest_sha256"] != plan["source_manifest_sha256"] or frozen["source_manifest_sha256"] != sha(files["source-manifest.json"]) or frozen["selection_sha256"] != sha(files["selection.json"]):
        raise ValueError("frozen candidate, source or validation selection changed")
    selection = strict_json(files["selection.json"])
    if selection["selection_uses_audit"] is not False:
        raise ValueError("original selection used audit")
    history = strict_json(files["F3-validation-history.json"])
    eligible = [row for row in history if not eligibility(row["formats"], frozen["untrained"]["formats"], selection["shortcuts"])]
    if not eligible or min(eligible, key=support()["core"].rank_key) != frozen["candidate"]:
        raise ValueError("original deterministic validation selection changed")
    restoration = strict_json(files["F3-restoration.json"])
    if restoration["parameter_bytes_equal"] is not True or restoration["validation_replay_equal"] is not True or strict_json(files["dataset-gate.json"])["ok"] is not True or strict_json(files["F3-dataset-gate.json"])["ok"] is not True:
        raise ValueError("original restoration/source checks failed")
    access = strict_json(files["audit-access.json"])
    if access["frozen_candidate_sha256"] != frozen["candidate"]["checkpoint_sha256"]:
        raise ValueError("original audit accessed a different candidate")
    coefficients = strict_json(files["baseline-coefficients.json"])
    if sha(files["baseline-coefficients.json"]) != EXPERIMENT["warm_start"]["coefficients_sha256"] or sha(files["baseline-coefficients.json"]) != strict_json(files["source-manifest.json"])["coefficient_sha256"]:
        raise ValueError("training-only fitted coefficients changed")
    for name in (FAMILY + ".safetensors", "UNTRAINED.safetensors"):
        package = files[name]
        length = struct.unpack("<Q", package[:8])[0]
        if not 0 < length <= 1 << 20 or length + 8 >= len(package):
            raise ValueError("invalid bounded package header")
        metadata = strict_json(package[8:8+length])["__metadata__"]
        licences = strict_json(files["licences.json"])
        for key, original in (("x-upstream-notice", "notice"), ("x-upstream-apache-2.0", "apache_2_0"), ("x-upstream-mit", "kit_mit"), ("x-licence", "sand")):
            if metadata.get(key) != licences[original]:
                raise ValueError("original package grant lost")
        parser = strict_json(files[name + ".parser.json"])
        if sha(package) != parser["sha256"] or len(package) != parser["bytes"] or parser["manifest"]["verification"] != {"stage1": True, "loaded": False}:
            raise ValueError("original exact parser/package binding changed")
    streams = strict_json(files[FAMILY + ".safetensors.score-bits.json"])
    log = files[FAMILY + ".safetensors.runtime.log"].decode()
    manifest = strict_json(files[FAMILY + ".safetensors.parser.json"])
    if "keccak256 " + manifest["keccak256"] not in log or any(value not in log for value in FOUNDATION.values() if value != FOUNDATION["foundation_engine.wasm"]):
        raise ValueError("original engine/package/tokenizer log identity changed")
    actual = [int(match.group(1)) for match in re.finditer(r"item \d+: tokens_len=\d+ -> score=(\d+) \(0x[0-9a-f]{8}\)", log)]
    offset = 0
    for name in FORMATS:
        if len(streams[name]) != 512 or actual[offset:offset+512] != streams[name] or actual[offset+512:offset+1024] != streams[name]:
            raise ValueError("complete original repeated reference streams differ")
        for value in streams[name]:
            bits(value)
        offset += 1024
    if len(actual) != offset + 16:
        raise ValueError("original complete trained execution missing")
    return frozen, streams, coefficients


def prepare(root):
    remote_only()
    plan = read(HERE / "recovery.json")
    validate_plan(plan)
    request = urllib.request.Request(plan["url"], headers={"User-Agent": "CONTENT_MODERATION-recovery/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.status != 200:
            raise ValueError("original public artifact unavailable")
        raw = response.read((90 << 20) + 1)
    files = archive_members(raw, plan)
    verify_origin(files, plan)
    for name, payload in files.items():
        (root / "evidence" / name).write_bytes(payload)
    (root / "evidence/origin-trained.runtime.log").write_bytes(files[FAMILY + ".safetensors.runtime.log"])
    write(root / "evidence/origin-artifact-binding.json", plan)
    m = support()
    m["core"].load_shared()[1].prepare(root)
    compile_tools(root)
    capture_class(root)
    (root / "originals").mkdir()
    regenerated = m["freeze_data"].freeze(root, json.loads(os.environ.get("CAMPAIGN_EXCLUSIONS_JSON", "[]")))
    original = strict_json(files["audit-freeze.json"])
    for key in ("audit_sha256", "audit_groups", "audit_labels", "exclusion_sha256", "original_files", "extra_exclusions_sha256"):
        if regenerated[key] != original[key]:
            raise ValueError("byte-preserving source recovery changed: " + key)
    # Preserve the original frozen source report, not a new report or fitted data.
    (root / "evidence/audit-freeze.json").write_bytes(files["audit-freeze.json"])


def parse_exact(root, name):
    package = root / "evidence" / name
    spec = read(root / "evidence/foundation.json")
    spec["base_model"] = str(root / "base")
    write(root / "evidence/recovery-foundation.json", spec)
    capture_class(root)
    command = [str(root / "stagecheck"), "--intent", FAMILY, "--child", FAMILY + "_GENERAL", "--expected-class", "judgment",
               "--package", str(package), "--sha256", sha(package.read_bytes()), "--foundation", str(root / "evidence/recovery-foundation.json"),
               "--foundation-sha256", sha((root / "evidence/recovery-foundation.json").read_bytes()), "--class-snapshot", str(root / "evidence/class-snapshot.json"),
               "--snapshot-sha256", sha((root / "evidence/class-snapshot.json").read_bytes()), "--out", str(root / "evidence")]
    result = subprocess.run(command, capture_output=True, timeout=60, check=False)
    (root / "evidence" / (name + ".recovery.parser.stdout")).write_bytes(result.stdout[:1 << 20])
    (root / "evidence" / (name + ".recovery.parser.stderr")).write_bytes(result.stderr[:1 << 20])
    if result.returncode or len(result.stdout) > 1 << 20 or len(result.stderr) > 1 << 20:
        raise ValueError("actual recovery full parser failed")
    parsed = strict_json(result.stdout)
    original = read(root / "evidence" / (name + ".parser.json"))
    if any(parsed[key] != original[key] for key in ("bytes", "sha256", "keccak256")) or parsed["manifest"]["verification"] != {"stage1": True, "loaded": False}:
        raise ValueError("recovery parser changed package identity")
    write(root / "evidence" / (name + ".recovery.parser.json"), parsed)
    return package, parsed


def measure(root):
    remote_only()
    plan = read(HERE / "recovery.json")
    files = {row["name"]: (root / "evidence" / row["name"]).read_bytes() for row in plan["files"]}
    for row in plan["files"]:
        if sha(files[row["name"]]) != row["sha256"]:
            raise ValueError("recovered origin input changed")
    frozen, trained_bits, coefficients = verify_origin(files, plan)
    m = support()
    torch, np, trainer, hp = m["model_ops"].runtime(root, EXPERIMENT["epochs"])
    # Imports supply the pinned renderer/tokenizer only. No model is built or fit.
    del torch, np, hp
    tok, node = m["model_ops"].tokenizer(root)
    raw = (root / "audit-private/audit.jsonl").read_bytes()
    if sha(raw) != plan["audit_sha256"]:
        raise ValueError("original independent audit changed")
    data, _, _ = m["core"].load_shared()
    originals = data.decode_jsonl(raw)
    if len(originals) != 256 or len({r["group_id"] for r in originals}) != 256:
        raise ValueError("original complete audit coverage changed")
    data.check_splits({**m["model_ops"].read_development(root), "audit": originals})
    views = m["model_ops"].views(originals)
    m["model_ops"].tokenize(trainer, tok, node, list(views.values()), root / "evidence/recovery-tokenizer-check.json")
    shortcuts = {name: max(shortcut_scores(rows, coefficients).values()) for name, rows in views.items()}
    token_vectors = {name: [node.encode(trainer.render(r["prompt"], r[side])).ids for r in rows for side in ("chosen", "rejected")] for name, rows in views.items()}
    trained_package, trained_manifest = parse_exact(root, FAMILY + ".safetensors")
    untrained_package, untrained_manifest = parse_exact(root, "UNTRAINED.safetensors")
    _, vectors, negative = engine_audit(root, trained_package, trained_manifest, {})
    untrained_bits, _, _ = engine_audit(root, untrained_package, untrained_manifest, token_vectors)
    binding = {"kind": "Byte-preserving continuation of the original interrupted joint audit, no retraining or selection",
               "origin_run_id": plan["origin_run_id"], "origin_commit": plan["origin_commit"], "origin_artifact_id": plan["artifact_id"],
               "origin_archive_sha256": plan["archive_sha256"], "origin_source_manifest_sha256": plan["source_manifest_sha256"],
               "original_trained_repeated_scores": 3072, "audit_sha256": plan["audit_sha256"],
               "recovery_run_id": int(os.environ["GITHUB_RUN_ID"]), "recovery_commit": os.environ["GITHUB_SHA"],
               "recovery_plan_sha256": sha((HERE / "recovery.json").read_bytes())}
    finish_report(root, frozen, trained_package, trained_manifest, originals, views, trained_bits, untrained_bits, shortcuts, vectors, negative, binding)


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("run", "prepare", "measure"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase != "run":
        {"prepare": prepare, "measure": measure}[args.phase](root)
        return
    if root.exists():
        raise ValueError("fresh runtime recovery directory required")
    (root / "evidence").mkdir(parents=True)
    deadline = time.monotonic() + 15000
    command = [sys.executable, str(HERE / "recover.py")]
    try:
        execute_phase(root, "recovery-prepare", [*command, "prepare", "--root", str(root)], 1800, deadline)
        execute_phase(root, "recovery-measure", [*command, "measure", "--root", str(root)], 12600, deadline)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
        write(root / "evidence/operational-failure.json", {"status": "FAILED", "error_type": type(error).__name__, "release_gate": "INCOMPLETE", "model_bytes_changed": False})
        raise


if __name__ == "__main__":
    main()
