# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote byte-preserving integration recovery, no fitting or audit rescoring."""
import argparse
import os
import shutil
import signal
from pathlib import Path

from data import FAMILY, HERE, fetch, write_json
from frozen_candidate import original_evidence, verify_frozen_package
from licensing import SIDECARS, licence_metadata
from reference import RELEASE, prepare_kit
from report import registration_ready
from runtime_check import (
    BATCH,
    RELEASE_HASHES,
    compile_tools,
    model_remote_only,
    server,
    validate_package,
)


def preserve_grants(root):
    root.mkdir(parents=True, exist_ok=True)
    for name in SIDECARS:
        shutil.copyfile(HERE / name, root / name)
    write_json(root / "licence-metadata.json", licence_metadata())


def run(root):
    model_remote_only()
    preserve_grants(root)
    freeze, report, primary = original_evidence()
    package = root / (FAMILY + ".safetensors")
    shutil.copyfile(HERE / "frozen/TEXT_SUMMARIZATION.safetensors", package)
    verify_frozen_package(package)
    shutil.copyfile(HERE / "frozen/foundation.json", root / "foundation.json")
    kit = prepare_kit(root)
    compile_tools(root, kit)
    foundation = root / "foundation"
    foundation.mkdir(exist_ok=True)
    for name, sha in RELEASE_HASHES.items():
        fetch(RELEASE + name, foundation / name, sha)
    checked = validate_package(root)
    if checked["sha256"] != freeze["package_sha256"] or checked["keccak256"].removeprefix("0x") != freeze["package_keccak"]:
        raise ValueError("fresh full parser differs from the original package identity")
    expected = report["runtime"]["vectors"]
    tokens = [v["tokens"] for v in expected]
    wanted = [v["score_bits"] for v in expected]
    with server(root, checked, "recovery-first") as score:
        first = [b for i in range(0, len(tokens), BATCH) for b in score(tokens[i:i + BATCH])]
        repeated = [b for i in range(0, len(tokens), BATCH) for b in score(tokens[i:i + BATCH])]
        negative = 0
        for key in ("foundation_hash", "tokenizer_hash", "adapter_hash"):
            try:
                score(tokens[:1], {key: "0" * 64})
            except ValueError:
                negative += 1
            else:
                raise ValueError("fresh negative identity escaped refusal")
        for bad in ([], [-1]):
            score([bad], refusal=True)
    with server(root, checked, "recovery-fresh") as score:
        fresh = [b for i in range(0, len(tokens), BATCH) for b in score(tokens[i:i + BATCH])]
    if first != wanted or repeated != wanted or fresh != wanted:
        raise ValueError("fresh engine vectors differ from original immutable package")
    runtime = {**report["runtime"], "vectors": [{"tokens": t, "score_bits": b} for t, b in zip(tokens, first)],
               "wrong_identity_rejections": negative, "repeat_bits_identical": first == repeated == fresh,
               "package_sha256": checked["sha256"], "adapter_hash": checked["keccak256"]}
    origin = {"run_id": freeze["training_run_id"], "commit": freeze["training_commit"],
              "artifact_id": freeze["training_artifact_id"], "archive_sha256": freeze["training_archive_sha256"],
              "runtime_report_sha256": next(f["sha256"] for f in freeze["files"] if f["path"] == "frozen/runtime-result.json"),
              "result_report_sha256": next(f["sha256"] for f in freeze["files"] if f["path"] == "frozen/result.json")}
    recovered = {**report, "runtime": runtime, "package_manifest": checked}
    ready = registration_ready(root, recovered, primary, training_origin=origin)
    write_json(root / "recovery-result.json", {"status": "PASS", "SPDX-License-Identifier": "LicenseRef-zkasuran-SAND-1.0",
               "family": FAMILY, "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
               "original_training": origin, "package_sha256": checked["sha256"], "package_keccak": freeze["package_keccak"],
               "fitting_performed": False, "audit_rescored": False, "full_parser_passed": True,
               "fixed_vectors": len(first), "negative_identity_controls": negative, "repeat_bits_identical": True,
               "node_qualification": "not attempted", "ready_status": ready["status"]})


if __name__ == "__main__":
    model_remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if not args.root.resolve().is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("recovery output must be within the bounded public runner")
    def timeout(signum, frame):
        raise TimeoutError("bounded integration recovery timed out")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(80 * 60)
    try:
        run(args.root)
    finally:
        preserve_grants(args.root)
