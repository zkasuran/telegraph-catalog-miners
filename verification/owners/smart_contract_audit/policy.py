# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pure admission, quality and remote-authority checks."""
import os
import re

REPO = "zkasuran/telegraph-catalog-miners"


def public_repository(event, context_repository):
    """Use the hosted workflow's real event metadata, not a rate-limited lookup."""
    repository = event.get("repository", {}) if isinstance(event, dict) else {}
    if not isinstance(repository, dict) or context_repository != REPO or repository.get("full_name") != REPO or repository.get("private") is not False:
        raise ValueError("exact owned public repository metadata required")
    return True


def remote_only():
    expected = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "zkasuran/telegraph-catalog-miners",
                "RUNNER_ENVIRONMENT": "github-hosted", "RUNNER_OS": "Linux"}
    if any(os.environ.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Model loading, training and inference require the controller's standard public CPU workflow")
    if os.environ.get("GITHUB_WORKFLOW") != "SMART_CONTRACT_AUDIT owner pipeline":
        raise RuntimeError("unexpected remote workflow")
    if not os.environ.get("GITHUB_RUN_ID") or not os.environ.get("GITHUB_SHA"):
        raise RuntimeError("missing run identity")
    if not re.fullmatch(r"[a-f0-9]{32}", os.environ.get("OWNER_RUN_ID", "")):
        raise RuntimeError("missing current owner attempt identity")


def execution_gate(rows, report, inventory_sha):
    if report.get("outcome") != "pass" or report.get("inventory_sha256") != inventory_sha:
        raise ValueError("execution report missing, incomplete or wrong inventory")
    measured = report.get("rows", [])
    if len(measured) != len(rows) or len({r["id"] for r in measured}) != len(rows):
        raise ValueError("execution coverage")
    indexed = {r["id"]: r for r in measured}
    for row in rows:
        r = indexed.get(row["id"], {})
        if r.get("source_sha256") != row["source_sha256"] or r.get("accepted_label") != row["vulnerable"] or r.get("assertions") != "pass":
            raise ValueError("source or label not execution-bound")
        bad, good = r["observed"]["bad"], r["observed"]["normal"]
        if bad["reverted"] == row["vulnerable"] or good["reverted"]:
            raise ValueError("incorrect execution outcomes")


def quality(metrics):
    failures = []
    if set(metrics) != {"prose", "json", "report"}:
        failures.append("missing format")
    for name, m in metrics.items():
        if m["accuracy"] < 0.80:
            failures.append(name + " accuracy below 0.80")
        if m["accuracy"] < m["untrained_accuracy"] + 0.05:
            failures.append(name + " improvement below 0.05")
        if min(m["by_mechanism"].values()) < 0.70 or min(m["by_label"].values()) < 0.70:
            failures.append(name + " subgroup below 0.70")
        if m.get("answer_only_accuracy", 1) > 0.65:
            failures.append(name + " answer-only shortcut")
        if m["accuracy"] < m.get("shortcut_accuracy", 1) + 0.10:
            failures.append(name + " shortcut improvement below 0.10")
    return {"verdict": "FAIL" if failures else "PASS", "failures": failures}


def main():
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--check-public-contract", action="store_true")
    args = parser.parse_args()
    if args.check_public_contract:
        public_repository({"repository": {"full_name": REPO, "private": False}}, REPO)
        for value in (True, None, 0, "false"):
            try:
                public_repository({"repository": {"full_name": REPO, "private": value}}, REPO)
            except ValueError:
                pass
            else:
                raise ValueError("private/ambiguous repository control passed")
        print(json.dumps({"public_repository_positive_and_refusal_controls": 5, "neural_work": False}))
        return
    remote_only()
    if os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch" or args.event is None or str(args.event) != os.environ.get("GITHUB_EVENT_PATH"):
        raise ValueError("actual hosted dispatch event path required")
    raw = args.event.read_bytes()
    if len(raw) > 1 << 20:
        raise ValueError("event metadata ceiling")
    event = json.loads(raw)
    passed = public_repository(event, os.environ["GITHUB_REPOSITORY"])
    report = {"public_repository_verified": passed, "repository": REPO, "run_id": os.environ["GITHUB_RUN_ID"],
              "evidence": "GitHub hosted workflow event payload and immutable default repository context"}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
