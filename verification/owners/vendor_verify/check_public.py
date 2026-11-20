# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only package and owned workflow contract checks."""

import json
from pathlib import Path

CHECKOUT = "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683"
UPLOAD = "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
WORKFLOW = ".github/workflows/evaluator-owner-vendor_verify.yml"


def validate_workflow(value):
    if not isinstance(value, dict) or value.get("permissions") != {"contents": "read"}:
        raise ValueError("read-only workflow permissions required")
    jobs = value.get("jobs", {})
    if set(jobs) != {"source_checks"}:
        raise ValueError("one bounded source-only job required")
    job = jobs["source_checks"]
    if job.get("runs-on") != "ubuntu-24.04" or job.get("timeout-minutes") != 10:
        raise ValueError("free public CPU runner and ten-minute bound required")
    steps = job.get("steps", [])
    if (
        len(steps) != 4
        or steps[0].get("uses") != CHECKOUT
        or steps[3].get("uses") != UPLOAD
    ):
        raise ValueError("pinned checkout/upload steps required")
    if steps[0].get("with", {}).get("persist-credentials") is not False:
        raise ValueError("persisted checkout credentials refused")
    if (
        steps[1].get("run")
        != "python3 -m pip install --disable-pip-version-check --no-input ruff==0.15.18"
    ):
        raise ValueError("exact source linter dependency required")
    if (
        steps[2].get("run")
        != "python3 -W error verification/owners/vendor_verify/ci.py --out source-proof"
    ):
        raise ValueError("only the actual packaged source executor is supported")
    dispatch = value.get("on", {}).get("workflow_dispatch", {}).get("inputs", {})
    if (
        set(dispatch) != {"owner_attempt"}
        or dispatch["owner_attempt"].get("type") != "string"
    ):
        raise ValueError("explicit attempt input required")
    if any(
        word in json.dumps(value).lower()
        for word in ("self-hosted", "secrets.", "workflow_run", "pull_request_target")
    ):
        raise ValueError("unsupported runner/privilege trigger")
    return True


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    raw = (root / WORKFLOW).read_bytes()
    if len(raw) > 65536:
        raise ValueError("workflow byte ceiling")
    validate_workflow(json.loads(raw.decode().split("\n", 1)[1]))
    for path in here.glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    for path in here.glob("*.json"):
        json.loads(path.read_bytes())
    result = json.loads((here / "SOURCE-RESULT.json").read_bytes())
    if result["status"] != "BLOCKED_FOR_TRAINING" or result["model_loading"]:
        raise ValueError("source-only result cannot claim model qualification")
    target = json.loads((here / "TASK-SCOPE.json").read_bytes())
    if (
        target["observation_task"] != "Norwegian domestic VAT observation consistency"
        or target["vies_truth"] is not False
    ):
        raise ValueError("domestic/VIES scope changed")
    print("Workflow, source-result, scope and actual Python compilation passed")


if __name__ == "__main__":
    main()
