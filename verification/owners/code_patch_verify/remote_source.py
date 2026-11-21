# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Bounded public CPU source smoke. These process results are not CI run labels."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from remote import remote_only
from export_record import measured_training_record


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("source proof must use runner temporary storage")
    args.out.mkdir(exist_ok=True)
    source_root = Path(os.environ["RUNNER_TEMP"]) / "cpv-source-kit"
    source_root.mkdir(exist_ok=True)
    pins = json.loads((Path(__file__).resolve().parent / "runtime_assets/sources.json").read_text())["kit"]
    archive = source_root / "kit.tar.gz"
    with urllib.request.urlopen("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + pins["revision"], timeout=30) as response:
        raw = response.read((32 << 20) + 1)
    if len(raw) > 32 << 20 or hashlib.sha256(raw).hexdigest() != pins["sha256"]:
        raise ValueError("primary kit archive identity/ceiling mismatch")
    archive.write_bytes(raw)
    with tarfile.open(archive) as source:
        if sum(member.size for member in source.getmembers()) > 128 << 20:
            raise ValueError("primary source archive expansion ceiling")
        source.extractall(source_root, filter="data")
    kit = source_root / ("telegraph-wasm-baseline-" + pins["revision"])
    wire = subprocess.run(["go", "test", "-mod=readonly", "-v", str(Path(__file__).resolve().parent / "wirecheck_test.go")],
                          cwd=kit, capture_output=True, timeout=600)
    (args.out / "wirecheck.stdout.txt").write_bytes(wire.stdout)
    (args.out / "wirecheck.stderr.txt").write_bytes(wire.stderr)
    if wire.returncode:
        raise ValueError("actual primary-source Go wire/boolean checks failed")
    # Synthetic source controls only, never a training record for a real model.
    controls = args.out / "training-record-CONTROL-ONLY.json"
    controls.write_text(json.dumps({
        "fixture_authorship": "synthetic source control, not model or audit data",
        "spdx": "LicenseRef-zkasuran-SAND-1.0",
        "original": {"family": "CODE_PATCH_VERIFY", "preference_authorship": "original constructed AI-assisted fixtures"},
        "fixed": measured_training_record([[1.0, 0.0], [1.0, 0.0]], "a" * 64, audit_used_for_selection=False)
    }, sort_keys=True) + "\n")
    record_env = {**os.environ, "CPV_SOURCE_RECORD_CONTROLS": str(controls)}
    record = subprocess.run(["go", "test", "-mod=readonly", "-v", str(Path(__file__).resolve().parent / "recordcheck_test.go")],
                            cwd=kit, env=record_env, capture_output=True, timeout=600)
    (args.out / "recordcheck.stdout.txt").write_bytes(record.stdout)
    (args.out / "recordcheck.stderr.txt").write_bytes(record.stderr)
    if record.returncode:
        raise ValueError("actual node training-record contract controls failed")
    results = []
    for name, program, expected in (("healthy-control", "assert 2 + 2 == 4\n", 0),
                                    ("deliberate-failure-control", "assert 2 + 2 == 5\n", 1)):
        path = args.out / (name + ".py")
        path.write_text("# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0\n" + program)
        completed = subprocess.run([sys.executable, "-I", str(path)], capture_output=True, timeout=10)
        if completed.returncode != expected:
            raise ValueError("original source process control failed")
        results.append({"source_file": path.name, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "exit_code": completed.returncode, "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
                        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest()})
    report = {"schema_version": 1, "family": "CODE_PATCH_VERIFY", "status": "SOURCE_EVIDENCE_ONLY",
              "run_id": int(os.environ["GITHUB_RUN_ID"]), "run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
              "commit": os.environ["GITHUB_SHA"], "runner_os": os.environ["RUNNER_OS"], "runner_arch": os.environ["RUNNER_ARCH"],
              "process_results": results, "training_permitted": False, "protected_rows_used": False,
              "model_operations": "not_attempted", "reference_engine": "not_executed", "registration_ready": False,
              "source_only_wire_tests_passed": True, "source_only_training_record_controls_passed": True,
               "training_record_controls_are_synthetic": True, "kit_revision": pins["revision"], "kit_archive_sha256": pins["sha256"],
              "scope": "Own measured process smokes, not workflow run conclusion, semantic repair or independently qualified model data",
              "next_gate": "Genuine trusted CI evidence delivery in the literal deployed scorer prompt"}
    (args.out / "source-evidence.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
