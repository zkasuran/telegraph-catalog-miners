# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Execute the source gate and report actual run identities, never model readiness."""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
from pathlib import Path


def identity(env):
    if env.get("GITHUB_ACTIONS") != "true":
        return {"mode": "local_source_checks", "run_id": None, "commit": None}
    if (
        env.get("GITHUB_REPOSITORY") != "zkasuran/telegraph-catalog-miners"
        or not re.fullmatch(r"[1-9][0-9]{0,19}", env.get("GITHUB_RUN_ID", ""))
        or not re.fullmatch(r"[0-9a-f]{40}", env.get("GITHUB_SHA", ""))
        or not re.fullmatch(r"[0-9a-f]{32}", env.get("OWNER_ATTEMPT", ""))
    ):
        raise ValueError(
            "actual remote repository/run/commit/attempt identities required"
        )
    return {
        "mode": "github_source_checks",
        "run_id": int(env["GITHUB_RUN_ID"]),
        "commit": env["GITHUB_SHA"],
        "owner_attempt": env["OWNER_ATTEMPT"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    ids = identity(os.environ)
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    out = args.out.resolve()
    if out.exists() or not out.is_relative_to(root) or out.is_relative_to(here):
        raise ValueError("new non-source output directory inside checkout required")
    out.mkdir(mode=0o700, parents=True)
    result = subprocess.run(
        ["bash", str(here / "verify.sh")],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    log = (result.stdout + result.stderr).encode()
    if len(log) > 2 << 20:
        raise ValueError("gate output ceiling")
    (out / "source-gate.log").write_bytes(log)
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(here.iterdir())
        if p.is_file()
    }
    report = {
        "schema_version": 1,
        "family": "VENDOR_VERIFY",
        **ids,
        "status": "SOURCE_CHECKS_PASSED"
        if result.returncode == 0
        else "SOURCE_CHECKS_FAILED",
        "returncode": result.returncode,
        "source_files_sha256": files,
        "workflow_sha256": hashlib.sha256(
            (root / ".github/workflows/evaluator-owner-vendor_verify.yml").read_bytes()
        ).hexdigest(),
        "gate_log_sha256": hashlib.sha256(log).hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "model_operations": False,
        "audit_evaluated": False,
        "registration_ready": False,
        "scope": "Source mechanics only, not neural accuracy or node qualification",
    }
    (out / "source-checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
