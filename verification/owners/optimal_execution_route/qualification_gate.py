# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Check actual source sufficiency. Never invent a model-ready verdict."""
import argparse
import json
from pathlib import Path

from oracle import Oracle, state_identity

HERE = Path(__file__).resolve().parent


def assess(root=HERE):
    task = json.loads((root / "primary/live-task.json").read_text())
    policy = json.loads((root / "policy.json").read_text())
    states = []
    for name in ("STATE.json", "STATE-OLDER.json"):
        states.extend(json.loads((root / "evidence" / name).read_text())["states"])
    groups = len({state_identity(s) for s in states})
    for state in states:
        Oracle(policy, state).verify_router_observations()
    inputs = [item["name"] for item in task["request"]["params"]]
    has_state = {"block_hash", "reference_state", "objective"} <= set(inputs)
    failures = []
    if not has_state:
        failures.append("Literal live request has no controller-trusted block/reference/objective fields")
    if groups < 32:
        failures.append("Retained regression states provide fewer than 32 independent groups")
    failures.append("No clean training/validation/frozen-candidate independent audit corpus or model package exists")
    return {"schema_version": 1, "family": "OPTIMAL_EXECUTION_ROUTE", "status": "QUALIFICATION_BLOCKED",
            "live_class": task["class"], "input_fields": inputs, "trusted_state_fields_present": has_state,
            "regression_state_groups": groups, "training_rows": 0, "audit_pairs": 0,
            "registration_ready": False, "reasons": failures,
            "alternatives_examined": [
                {"route": "bare numeric model", "result": "Same task/answer bytes require opposite exact order at two observed states"},
                {"route": "miner-supplied state", "result": "Untrusted answer evidence does not establish reference truth or freshness"},
                {"route": "input_root", "result": "Primary handler hashes received token batches, not independent blockchain state"},
                {"route": "ground_truth_value comparator", "result": "Available on deterministic wire, not assigned judgment class"},
                {"route": "contextual source preferences", "result": "Useful separate reasoning task, not proof for the bare live input"}],
            "scope": "Source-only blocker and regression evidence. No neural operation or canonical claim."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    report = assess()
    print(json.dumps(report, sort_keys=True))
    return 2 if args.require_ready and not report["registration_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
