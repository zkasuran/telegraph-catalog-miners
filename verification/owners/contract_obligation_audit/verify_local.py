# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Local source/contract verification. No model loading, training or inference."""
import ast
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from baselines import validate_frozen
from balanced import construct
from contracts import HERE, check_config, file_sha, sha, strict_json
from data import download
from source_gate import verdict
from parent_checkpoint import PARENT_RUN
from source_cues import measure_source_cues
from quality_selection import TRAINER_SHA256, check_experiments, compile_selector
from hans_curriculum import prepare as prepare_curriculum, require_reconstruction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-generation", type=int, required=True)
    args = parser.parse_args()
    if args.source_generation != 8:
        raise ValueError("retired source generations are not eligible releases")
    for path in HERE.glob("*.py"):
        raw = path.read_bytes()
        compile(raw, path.name, "exec")
        ast.parse(raw)
    config = check_config(strict_json((HERE / "config.json").read_bytes()))
    process = subprocess.run([sys.executable, "-W", "error", "-m", "unittest", "discover", "-s", str(HERE), "-p", "test_*.py", "-v"],
                             env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", "OWNER_SOURCE_ONLY": "1"},
                             capture_output=True, text=True, timeout=120)
    print(process.stdout + process.stderr, end="")
    if process.returncode:
        raise RuntimeError("source-only unit/primary-path gate failed")
    for path in HERE.glob("*.json"):
        strict_json(path.read_bytes())
    for name in ("LICENSE", "NOTICE", "CC-BY-4.0.txt", "DATA-SOURCES.md", "SECURITY.md"):
        if not (HERE/name).is_file() or not (HERE/name).stat().st_size:
            raise ValueError("missing retained source grant or threat model")
    policy = strict_json((HERE/"source-policy.json").read_bytes())
    fingerprints = strict_json((HERE/"source-fingerprints.json").read_bytes())
    if sha((HERE/"source-policy.json").read_bytes()) != fingerprints["policy_sha256"]:
        raise ValueError("source policy changed after the data freeze")
    cache = os.environ.get("OWNER_SOURCE_ARCHIVE")
    scratch_parent = Path("/tmp/opencode")
    scratch_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="contract-gate-", dir=scratch_parent) as temporary:
        trainer_cache = os.environ.get("OWNER_TRAINER_SOURCE")
        trainer_source = Path(trainer_cache) if trainer_cache else Path(temporary)/"train_adapter.py"
        if not trainer_cache:
            download("https://raw.githubusercontent.com/telegraphprotocol/telegraph-wasm-baseline/1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd/float_runtime/train_adapter.py",
                     trainer_source, TRAINER_SHA256, 128 << 10)
        compile_selector(trainer_source.read_bytes())  # Actual source compiled, no training executed.
        archive = Path(cache) if cache else Path(temporary)/"source.zip"
        if not cache:
            download(config["archive_url"], archive, config["archive_sha256"])
        if file_sha(archive) != config["archive_sha256"]:
            raise ValueError("source gate archive mismatch")
        splits, _ = construct(archive, config, policy)
        curriculum_cache = os.environ.get("OWNER_HANS_TRAINING")
        _, _, curriculum = prepare_curriculum(Path(temporary)/"curriculum", Path(curriculum_cache) if curriculum_cache else None)
        require_reconstruction(curriculum, curriculum)  # Required source fingerprint fields and primary acceptance path.
        curriculum_hashes = set(curriculum["selected_normalized_premise_sha256"])
        if any(row["normalized_evidence_sha256"] in curriculum_hashes for rows in splits.values() for row in rows):
            raise ValueError("auxiliary curriculum overlaps the grouped legal source")
        if file_sha(HERE/"HANS-MIT.txt") != "ece363d4ea0a01bc3dbe0b7fba5ff6c8d81d5866fea8948a1f353642cdb750fa":
            raise ValueError("original full HANS MIT notice changed")
        retirement = HERE/"retired-source-groups.json"
        if retirement.exists():
            retired = strict_json(retirement.read_bytes())
            if any(row["source_group"] in retired["groups"] for rows in splits.values() for row in rows):
                raise ValueError("archival source generation includes permanently retired audit groups; a fresh dataset is required")
        for name, rows in splits.items():
            raw = b"".join(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()+b"\n" for row in rows)
            if sha(raw) != fingerprints["files"][name]["sha256"]:
                raise ValueError("constructed source split differs from frozen source bytes")
        frozen = strict_json((HERE/"frozen-baselines-v8.json").read_bytes())
        validate_frozen(frozen, splits["train"])
        cues = strict_json((HERE/"frozen-source-cues-v8.json").read_bytes())
        if cues["training_data_sha256"] != fingerprints["files"]["train"]["sha256"]:
            raise ValueError("semantic coefficients are not bound to training data")
        for split in ("train", "validation"):
            measure_source_cues(splits[split], cues)
        checked = verdict(splits, frozen, config)
        if not checked["passed"]:
            raise ValueError("actual source gate failed in the standalone checkout")
    workflow = HERE.parents[2]/".github/workflows/evaluator-owner-contract_obligation_audit.yml"
    value = strict_json(workflow.read_bytes())  # JSON is a strict YAML 1.2 subset.
    check_experiments(value)
    jobs = value["jobs"]
    if "workflow_dispatch" not in value["on"] or not 1 <= len(jobs) <= 6:
        raise ValueError("owned workflow dispatch/job schema")
    if any(job["runs-on"] != "ubuntu-24.04" or not 1 <= job["timeout-minutes"] <= 350 for job in jobs.values()):
        raise ValueError("workflow runner/timeout ceiling")
    parent_steps = [step for step in jobs["train"]["steps"] if step.get("name") == "Read retained licensed best checkpoint"]
    if len(parent_steps) != 1 or parent_steps[0]["with"]["run-id"] != PARENT_RUN:
        raise ValueError("workflow does not download the exact retained checkpoint")
    print("ALL GREEN: source-only contracts, complete real source gate, immutable data/baselines and bounded workflow. No neural imports or operations.")


if __name__ == "__main__":
    main()
