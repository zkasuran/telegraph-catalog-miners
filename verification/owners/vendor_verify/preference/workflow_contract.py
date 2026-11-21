# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Literal bounded development-job contract, preserving the old source gate."""

import copy
import json

REMOTE_JOB = {
    "if": "${{ inputs.mode == 'train' }}",
    "runs-on": "ubuntu-24.04",
    "timeout-minutes": 350,
    "steps": [
        {
            "uses": "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683",
            "with": {"persist-credentials": False, "fetch-depth": 1},
        },
        {
            "name": "Install hash-pinned CPU dependencies",
            "run": "python3 -m pip install --disable-pip-version-check --no-input --only-binary=:all: --require-hashes --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu -r verification/owners/vendor_verify/preference/requirements.lock",
        },
        {
            "name": "Audit all pinned dependencies",
            "run": 'python3 verification/owners/vendor_verify/preference/audit_dependencies.py --lock verification/owners/vendor_verify/preference/requirements.lock --out "${{ runner.temp }}/vendor-audit-input.txt" --report "${{ runner.temp }}/vendor-dependency-audit.json"',
        },
        {
            "name": "Train real general preference model",
            "env": {
                "OWNER_ATTEMPT": "${{ inputs.owner_attempt }}",
                "PYTHONDONTWRITEBYTECODE": "1",
                "OPENBLAS_NUM_THREADS": "4",
            },
            "run": "python3 -W error verification/owners/vendor_verify/preference/remote.py",
        },
        {
            "uses": "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
            "if": "always()",
            "with": {
                "name": "vendor-general-development",
                "if-no-files-found": "error",
                "retention-days": 7,
                "path": "\n".join(
                    [
                        "${{ runner.temp }}/vendor-dependency-audit.json",
                        "${{ runner.temp }}/vendor-general-preference/development-result.json",
                        "${{ runner.temp }}/vendor-general-preference/development-failure.json",
                        "${{ runner.temp }}/vendor-general-preference/dataset-gate.json",
                        "${{ runner.temp }}/vendor-general-preference/tokenizer-check.json",
                        "${{ runner.temp }}/vendor-general-preference/training-filter.json",
                        "${{ runner.temp }}/vendor-general-preference/retained-tokenizer-check.json",
                        "${{ runner.temp }}/vendor-general-preference/filtered-dataset-gate.json",
                        "${{ runner.temp }}/vendor-general-preference/training-baseline-report.json",
                        "${{ runner.temp }}/vendor-general-preference/frozen-training-baseline.tar.gz",
                        "${{ runner.temp }}/vendor-general-preference/SELECTED-CHECKPOINT.json",
                        "${{ runner.temp }}/vendor-general-preference/selected-adapter.tar.gz",
                    ]
                ),
            },
        },
    ],
}
MODE = {
    "description": "Source contracts or real general preference development",
    "required": True,
    "type": "choice",
    "options": ["source", "train"],
    "default": "source",
}


def extend_workflow(old):
    value = copy.deepcopy(old)
    value["name"] = "Vendor source and general preference development"
    value["on"]["workflow_dispatch"]["inputs"]["mode"] = MODE
    value["jobs"]["source_checks"]["if"] = "${{ inputs.mode != 'train' }}"
    value["jobs"]["preference_training"] = copy.deepcopy(REMOTE_JOB)
    return value


def validate_extended_workflow(value, source_validator):
    if not isinstance(value, dict) or set(value.get("jobs", {})) != {
        "source_checks",
        "preference_training",
    }:
        raise ValueError("only source and one bounded preference job allowed")
    if value["jobs"]["preference_training"] != REMOTE_JOB:
        raise ValueError(
            "literal remote runner/timeout/dependency/upload contract changed"
        )
    if value.get("permissions") != {"contents": "read"}:
        raise ValueError("read-only permissions required")
    inputs = value.get("on", {}).get("workflow_dispatch", {}).get("inputs", {})
    if inputs.get("mode") != MODE or set(inputs) != {"mode", "owner_attempt"}:
        raise ValueError("exact mutually exclusive mode inputs required")
    original = copy.deepcopy(value)
    del original["jobs"]["preference_training"]
    del original["on"]["workflow_dispatch"]["inputs"]["mode"]
    if (
        original["jobs"]["source_checks"].pop("if", None)
        != "${{ inputs.mode != 'train' }}"
    ):
        raise ValueError("mutually exclusive source job required")
    if any(
        word in json.dumps(value).lower()
        for word in ("self-hosted", "secrets.", "pull_request_target", "workflow_run")
    ):
        raise ValueError("privilege/trigger escalation refused")
    # Execute the original validator against its exact projection, not a weaker
    # normalized source-only stand-in or a replacement of its tests.
    return source_validator(original)
