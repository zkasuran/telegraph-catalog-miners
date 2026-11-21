# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Standalone source-only release gate. It never imports a neural package."""
import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from build import build, check_splits
from common import sha, strict_json, write_json
from freeze import freeze
from groups import components
from scorer import normalized

HERE = Path(__file__).resolve().parent
WORKFLOW = HERE.parents[2] / ".github/workflows/evaluator-owner-event_outcome_resolution.yml"


def source_proof():
    frozen = strict_json((HERE / "frozen/frozen.json").read_bytes())
    splits = {}
    overlap = 0
    seen_people, seen_events, seen_prompts = set(), set(), set()
    for name in ("train", "validation", "audit"):
        raw = (HERE / "frozen" / (name + ".jsonl")).read_bytes()
        if sha(raw) != frozen["files"][name]["sha256"]:
            raise ValueError("frozen source byte mismatch")
        rows = [strict_json(l) for l in raw.splitlines()]
        people = {p for r in rows for p in r["people"]}
        events = {r["event_id"] for r in rows}
        prompts = {normalized(r["prompt"]) for r in rows}
        overlap += len(people & seen_people) + len(events & seen_events) + len(prompts & seen_prompts)
        seen_people |= people
        seen_events |= events
        seen_prompts |= prompts
        splits[name] = rows
    check_splits(splits)
    independent = components(splits["audit"])
    if overlap or independent < 32:
        raise ValueError("source overlap or independent audit groups failed")
    licences = strict_json((HERE / "licences.json").read_bytes())
    card = (HERE / "DATA-SOURCES.md").read_text()
    grants = ((HERE / "LICENSE").read_text() == licences["sand"] and
              "The Services are free to use according to the Creative Commons Zero (CC0) license." in card and
              "We release the ModernBERT model architectures, model weights, training codebase under the Apache 2.0 license." in card and
              "Permission is hereby granted, free of charge" in (HERE / "sources/kit-LICENSE").read_text() and
              (HERE / "NOTICE").stat().st_size > 100)
    if not grants:
        raise ValueError("source/model grant or retained licence missing")
    return {"normalized_overlap": overlap, "audit_used_for_selection": False, "grants_verified": grants,
            "source_manifest_sha256": sha((HERE / "frozen/frozen.json").read_bytes()), "independent_groups": independent,
            "audit_pairs": len(splits["audit"]), "source_receipts_sha256": frozen["source_receipts_sha256"]}


def config_gate():
    workflow = strict_json(WORKFLOW.read_bytes())
    if len(workflow["on"]["workflow_dispatch"]["inputs"]) > 10:
        raise ValueError("GitHub workflow dispatch input ceiling exceeded")
    if workflow["name"] != "Owner EVENT_OUTCOME_RESOLUTION" or workflow["permissions"] != {"contents": "read"}:
        raise ValueError("owned workflow authority/name mismatch")
    if workflow["concurrency"]["cancel-in-progress"] is not False or set(workflow["on"]) != {"workflow_dispatch"}:
        raise ValueError("workflow retry/cancellation scope mismatch")
    if len(workflow["jobs"]) > 6 or "secrets." in WORKFLOW.read_text() or "self-hosted" in WORKFLOW.read_text():
        raise ValueError("compute or credential scope mismatch")
    for job in workflow["jobs"].values():
        if job["runs-on"] != "ubuntu-24.04" or not 1 <= job["timeout-minutes"] <= 350:
            raise ValueError("paid or unbounded runner refused")
        if job.get("strategy", {}).get("max-parallel", 1) > 6:
            raise ValueError("parallel compute ceiling")
        for step in job["steps"]:
            if "uses" in step:
                import re
                if not re.fullmatch(r"actions/(checkout|setup-go|upload-artifact|download-artifact)@[0-9a-f]{40}", step["uses"]):
                    raise ValueError("unpinned action")
            if "run" in step:
                subprocess.run(["bash", "-n", "-c", step["run"]], check=True, timeout=10)


def main():
    for name in ("verify.sh", "LICENSE", "NOTICE", "DATA-SOURCES.md", "SECURITY.md"):
        if not (HERE / name).is_file() or not (HERE / name).stat().st_size:
            raise ValueError("required standalone publication asset missing")
    for path in HERE.glob("*.py"):
        compile(path.read_text(), str(path), "exec")
    config_gate()
    with tempfile.TemporaryDirectory(prefix="event-owner-gate-") as temp:
        root = Path(temp)
        build(HERE / "sources", root / "data")
        freeze(root / "data", root / "frozen")
        for folder in ("data", "frozen"):
            for path in (root / folder).iterdir():
                if path.read_bytes() != (HERE / folder / path.name).read_bytes():
                    raise ValueError("nonreproducible source/data artifact " + path.name)
    result = source_proof()
    import test_event
    mutant = types.ModuleType("mutant_event_scorer")
    exec(compile((HERE / "scorer.py").read_text().replace('return "unresolved"', 'return "no"'), "mutant-scorer", "exec"), mutant.__dict__)
    suite = unittest.TestSuite(test_event.ScorerTests(name) for name in ("test_unknown_is_not_negative", "test_same_day_unknown_time"))
    failure_control = unittest.TestResult()
    with patch.object(test_event, "resolve", mutant.resolve):
        suite.run(failure_control)
    if len(failure_control.failures) != 2 or failure_control.errors:
        raise ValueError("original controls did not detect actual unknown-to-no mutation")
    print("SOURCE_ONLY_GATE_PASS", json.dumps(result, sort_keys=True))
    print("Actual unknown-to-no scorer mutation: 2 expected failures")


if __name__ == "__main__":
    main()
