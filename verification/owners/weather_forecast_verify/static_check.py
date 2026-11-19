# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Actual syntax compilation, strict config checks and immutable bundle checks."""
import ast
import json
import re

from build import HERE, sha


def validate_workflow(spec):
    if spec["name"] != "Weather archive evaluator experiment" or spec["permissions"] != {"contents": "read"}:
        raise ValueError("workflow identity or permissions")
    if set(spec["on"]) != {"workflow_dispatch"} or spec["concurrency"]["cancel-in-progress"] is not False:
        raise ValueError("manual bounded dispatch and durable resumes required")
    if set(spec["jobs"]) != {"train", "audit", "serve"} or spec["jobs"]["audit"].get("needs") != "train":
        raise ValueError("independent audit after train required")
    for name, job in spec["jobs"].items():
        if job["runs-on"] != "ubuntu-24.04" or job["timeout-minutes"] != {"train": 160, "audit": 110, "serve": 330}[name]:
            raise ValueError("fixed standard public CPU budget")
        for step in job["steps"]:
            if "uses" in step and not re.fullmatch(r"actions/[a-z-]+@[a-f0-9]{40}", step["uses"]):
                raise ValueError("unpinned workflow dependency")
            text = json.dumps(step)
            if "secrets." in text or "self-hosted" in text or "sudo " in text:
                raise ValueError("credentials or nonstandard resources refused")
            if step.get("uses", "").startswith("actions/checkout@") and step["with"].get("persist-credentials") is not False:
                raise ValueError("checkout credentials must not persist")
    return True


def validate_contract(spec):
    if (spec["family"] != "WEATHER_FORECAST_VERIFY" or spec["class"] != "judgment"
            or spec["run_id"] != "cb65961105934a849aceec6d66add0db" or spec["source_evidence_in_prompt"] is not False):
        raise ValueError("exact task/attempt or input contract changed")
    hp = spec["hyperparameters"]
    if hp["epochs"] != 2 or hp["max_length"] != 256 or hp["lora_rank"] != 8 or hp["early_stopping"] is not False:
        raise ValueError("fixed experiment hyperparameters changed")
    if spec["quality"] != {"minimum_pair_accuracy_each_view": 0.80, "minimum_offset_direction_accuracy": 0.70,
                          "minimum_untrained_gain": 0.05, "minimum_frozen_shortcut_gain": 0.10, "maximum_frozen_shortcut_accuracy": 0.65}:
        raise ValueError("quality gate changed")
    return True


def main():
    count = 0
    for path in sorted(HERE.glob("*.py")):
        raw = path.read_bytes()
        compile(raw, str(path), "exec")
        tree = ast.parse(raw)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert) and path.name not in ("test_weather.py",):
                raise ValueError("production assertion must not disappear under -O")
        count += 1
    validate_workflow(json.loads((HERE / "workflow.json").read_text()))
    validate_contract(json.loads((HERE / "contract.json").read_text()))
    frozen = json.loads((HERE / "freeze.json").read_text())
    for path, expected in frozen["files"].items():
        if sha((HERE / path).read_bytes()) != expected:
            raise ValueError("immutable file changed: " + path)
    print(json.dumps({"syntax_compiled_files": count, "configs_validated": 2, "frozen_bindings": len(frozen["files"])}))


if __name__ == "__main__":
    main()
