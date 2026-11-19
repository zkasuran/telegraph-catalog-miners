# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Stage a closed set of public source proofs, never the hidden build directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_FILE_BYTES = 4 * 1024 * 1024
FILES = {
    ".gate-work/source-report.json": "source-report.json",
    ".gate-work/remote-source-report.json": "remote-source-report.json",
    ".gate-work/lint.log": "lint.log",
    ".gate-work/go-version.log": "go-version.log",
    ".gate-work/go-build.log": "go-build.log",
    ".gate-work/primary-wire.log": "primary-wire.log",
    ".gate-work/tests.log": "tests.log",
    ".gate-work/primary-smoke.log": "primary-smoke.log",
    ".gate-work/tamper-smoke.log": "tamper-smoke.log",
    ".gate-work/qualification-negative.log": "qualification-negative.log",
    "LICENSE": "LICENSE",
    "NOTICE": "NOTICE",
    "source-manifest.json": "source-manifest.json",
    "upstream/UNISWAP-MIT.txt": "UNISWAP-MIT.txt",
    "upstream/TELEGRAPH-MIT.txt": "TELEGRAPH-MIT.txt",
}


def stage(root=HERE, allow_partial=False):
    root = root.resolve()
    output = root / "result"
    if output.is_symlink():
        raise ValueError("linked result directory")
    output.mkdir(exist_ok=True)
    allowed = set(FILES.values()) | {"RESULT-MANIFEST.json"}
    if any(p.name not in allowed or p.is_symlink() or not p.is_file() for p in output.iterdir()):
        raise ValueError("unexpected staged payload; preserve it instead of uploading")
    rows, missing = [], []
    for name, target in FILES.items():
        source = root / name
        if not source.is_file():
            missing.append(name)
            continue
        if source.is_symlink() or not source.resolve().is_relative_to(root) or source.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("linked, escaped or oversized source proof")
        destination = output / target
        if destination.is_symlink():
            raise ValueError("linked staged proof")
        raw = source.read_bytes()
        destination.write_bytes(raw)
        rows.append({"path": target, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    report = {"schema_version": 1, "family": "OPTIMAL_EXECUTION_ROUTE",
              "spdx": "LicenseRef-zkasuran-SAND-1.0", "upstream_grants_retained": True,
              "files": rows, "complete": False, "missing": missing,
              "scope": "Source verification reports only, not model or registration-ready artifacts"}
    (output / "RESULT-MANIFEST.json").write_text(json.dumps(report, indent=2) + "\n")
    if missing:
        if allow_partial:
            return report
        raise ValueError("incomplete source proof staging")
    source_raw = (output / "source-report.json").read_bytes()
    source = json.loads(source_raw)
    remote = json.loads((output / "remote-source-report.json").read_text())
    if (source.get("status") != "SOURCE_GATE_PASSED"
            or remote.get("status") != "SOURCE_VERIFIED_NOT_REGISTRATION_READY"
            or source.get("registration_ready") is not False or remote.get("registration_ready") is not False
            or remote.get("source_report_sha256") != hashlib.sha256(source_raw).hexdigest()):
        raise ValueError("source/remote proof identity or scope mismatch")
    report["complete"] = True
    (output / "RESULT-MANIFEST.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    if os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_REPOSITORY") != "zkasuran/telegraph-catalog-miners":
        raise ValueError("source staging CLI is restricted to the public workflow")
    parser = argparse.ArgumentParser()
    parser.add_argument("--preserve-partial", action="store_true")
    report = stage(allow_partial=parser.parse_args().preserve_partial)
    if report["complete"]:
        remote = json.loads((HERE / "result/remote-source-report.json").read_text())
        if (str(remote.get("run_id")) != os.environ.get("GITHUB_RUN_ID")
                or remote.get("commit") != os.environ.get("GITHUB_SHA")
                or remote.get("owner_run_id") != os.environ.get("OWNER_RUN_ID")):
            raise ValueError("actual source run/commit/owner binding mismatch")
    print(json.dumps({"staged_files": len(report["files"]), "complete": report["complete"]}, sort_keys=True))


if __name__ == "__main__":
    main()
