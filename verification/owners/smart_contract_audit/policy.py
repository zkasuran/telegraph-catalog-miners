# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pure admission, quality and remote-authority checks."""
import os
import re


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
