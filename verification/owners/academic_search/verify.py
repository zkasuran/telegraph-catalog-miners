# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Source-only gate. No neural imports, models or hidden evaluation data."""
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from source import HERE, annotations, consensus, sha


def validate_config(config):
    from wire import FOUNDATION, ENGINE, TOKENIZER
    fields = {"schema_version", "family", "child", "class", "formats", "source_scope", "independent_originals", "hyperparameters", "selection", "quality", "maximum_workflow_minutes", "foundation_sha256", "engine_sha256", "tokenizer_sha256", "status"}
    if set(config) != fields or config["schema_version"] != 1:
        raise ValueError("exact model configuration schema required")
    if (config["family"] != "ACADEMIC_SEARCH" or config["child"] != "ACADEMIC_SEARCH_GENERAL" or config["class"] != "judgment"
            or config["formats"] != ["title", "json", "prose"] or not isinstance(config["source_scope"], str) or not config["source_scope"]
            or config["independent_originals"] != {"train": 128, "validation": 32, "audit": 64}
            or config["hyperparameters"] != {"max_length": 256, "lora_rank": 8, "lora_alpha": 32, "batch_size": 1, "grad_accum": 16, "epochs": 3, "seed": 42, "eval_batch": 1}
            or config["selection"] != {"split": "validation", "metric": "val_loss", "direction": "min", "tie": "earlier_epoch", "audit_after_selection": True}
            or config["quality"] != {"minimum_accuracy": .8, "minimum_per_label": .7, "minimum_untrained_gain": .05, "minimum_shortcut_gain": .1, "maximum_shortcut_accuracy": .65}
            or type(config["maximum_workflow_minutes"]) is not int or not 1 <= config["maximum_workflow_minutes"] <= 350
            or (config["foundation_sha256"], config["engine_sha256"], config["tokenizer_sha256"]) != (FOUNDATION, ENGINE, TOKENIZER)):
        raise ValueError("model configuration identity, split or quality gate mismatch")


def validate_workflow(value):
    if set(value.get("jobs", {})) != {"academic"} or value.get("permissions") != {"contents": "read"}:
        raise ValueError("one read-only standard public CPU job required")
    job = value["jobs"]["academic"]
    if job.get("runs-on") != "ubuntu-24.04" or type(job.get("timeout-minutes")) is not int or not 1 <= job["timeout-minutes"] <= 350 or "strategy" in job:
        raise ValueError("bounded ubuntu-24.04 job without fanout required")
    if value.get("concurrency", {}).get("cancel-in-progress") is not False:
        raise ValueError("healthy jobs must not be cancelled")
    inputs = value.get("on", {}).get("workflow_dispatch", {}).get("inputs", {})
    if inputs.get("mode", {}).get("options") not in (["train", "serve"], ["train", "serve", "develop"]):
        raise ValueError("actual train and verified-serving modes required")
    if not job.get("steps") or any(not isinstance(step, dict) for step in job["steps"]):
        raise ValueError("workflow steps missing")
    for step in job["steps"]:
        if "uses" in step and not __import__("re").fullmatch(r"actions/(checkout|setup-go|upload-artifact)@[0-9a-f]{40}", step["uses"]):
            raise ValueError("workflow action must be hash-pinned")
        if "timeout-minutes" in step and not 1 <= step["timeout-minutes"] <= 350:
            raise ValueError("step timeout ceiling")
    for step in job["steps"]:
        if step.get("name") == "Upload ONLY restricted public forwarding lease" and step.get("with", {}).get("path") != "${{ runner.temp }}/academic-serving/restricted-public-lease.json":
            raise ValueError("serving may upload only the restricted public lease")


def main():
    start = time.monotonic()
    evidence = HERE / "evidence"
    evidence.mkdir(exist_ok=True)
    syntax = sorted(HERE.glob("*.py"))
    for path in syntax:
        compile(path.read_bytes(), str(path), "exec")
        ast.parse(path.read_text(), filename=str(path))
    config = json.loads((HERE / "model-contract.json").read_bytes())
    validate_config(config)
    development = HERE / "development-contract.json"
    if development.is_file():
        from curriculum import validate_development
        validate_development(json.loads(development.read_bytes()))
        retirement = json.loads((HERE / "audit-retirement.json").read_bytes())
        if (retirement.get("retired_audit_sha256") != "75701820c853b3f1348ae930271779dfd253deff2b5a9f3230cb6fbae916eb5c"
                or retirement.get("later_qualification_requires_clean_independent_audit") is not True):
            raise ValueError("old exposed audit retirement boundary changed")
    workflow = HERE / "workflow.yml"
    if not workflow.exists():
        workflow = HERE.parents[2] / ".github/workflows/evaluator-owner-academic_search.yml"
    if workflow.exists():
        text = workflow.read_text()
        validate_workflow(json.loads("\n".join(line for line in text.splitlines() if not line.startswith("#"))))
    bindings = HERE / "dependency-bindings.json"
    if bindings.exists():
        expected = json.loads(bindings.read_bytes())
        for original, owned in (("training/requirements.lock", "requirements.lock"), ("training/audit.py", "dependency_audit.py")):
            if sha((HERE / owned).read_bytes()) != expected[original]:
                raise ValueError("reviewed shared source copy changed")
        lease = HERE.parents[1] / "registration/common.py"
        if lease.exists() and sha(lease.read_bytes()) != expected["registration/common.py"]:
            raise ValueError("reviewed actual public lease source changed")
    from checkpoint import licence_metadata
    licence_metadata()
    command = [sys.executable, "-W", "error", "-m", "unittest", "discover", "-s", str(HERE), "-p", "test_contracts.py", "-v"]
    completed = subprocess.run(command, cwd=HERE, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, timeout=60)
    log = completed.stdout + completed.stderr
    logpath = evidence / ("contracts-" + str(time.time_ns()) + ".log")
    logpath.write_bytes(log)
    sys.stdout.buffer.write(log)
    if completed.returncode:
        raise ValueError("source contract tests failed")
    smoke = json.loads((HERE / "smoke.json").read_bytes())
    from data import entities, formats, render
    components = entities(smoke["metadata"])
    if len({components.find(i) for i in smoke["ids"]}) != 3:
        raise ValueError("public smoke contains aliased papers")
    pool, report = consensus(smoke["annotation_projection"])
    seed, chosen, rejected = smoke["ids"]
    if chosen not in {r["pmid"] for r in pool[seed]["relevant"]} or rejected not in {r["pmid"] for r in pool[seed]["irrelevant"]}:
        raise ValueError("public original annotation smoke mismatch")
    original = smoke["metadata"][chosen]
    shaped = formats(original["title"], original["doi"])
    if json.loads(shaped["json"])["answer"] != original["title"] or render("q", shaped["title"]) != "### Task:\nq\n\n### Answer:\n" + original["title"]:
        raise ValueError("public source smoke changed original or literal rendering")
    if (HERE / "frozen/freeze.json").is_file():
        from scorer import load_freeze
        load_freeze(HERE / "frozen")  # Hash/group/provenance checks only, no audit metrics.
    counts = None
    if (HERE / "sources/RELISH_v1.json.gz").is_file():
        raw = (HERE / "sources/RELISH_v1.json.gz").read_bytes()
        _, counts = consensus(annotations(raw))
        if counts["source_rows"] != 3278 or counts["unique_seed_candidate_pairs"] != 191640 or counts["conflicting_pairs_excluded"] != 1756:
            raise ValueError("full source recount disagrees with captured inventory")
    report = {"status": "SOURCE_BUILD_PASS_MODEL_UNVERIFIED", "syntax_files": len(syntax), "test_command": command,
              "test_returncode": completed.returncode, "test_log": str(logpath.relative_to(HERE)), "test_log_sha256": sha(log),
              "config_sha256": sha((HERE / "model-contract.json").read_bytes()), "primary_source_smoke": "pass",
              "full_annotation_recount": counts, "neural_imports": False, "model_inference": False,
              "elapsed_seconds": time.monotonic() - start}
    (evidence / "build-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("ALL GREEN: source build only. No model quality or reference-runtime pass claimed.")


if __name__ == "__main__":
    main()
