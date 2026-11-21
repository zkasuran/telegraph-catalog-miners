# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Validate the standalone owned workflow and immutable source configuration."""
import json
import re
from source_gate import HERE, PINS


def validate():
    import yaml
    config = yaml.load((HERE / "controller-workflow.yml").read_text(), Loader=yaml.BaseLoader)
    actual = HERE.parents[2] / ".github/workflows/evaluator-owner-llm_output_evaluation.yml"
    if actual.exists() and actual.read_bytes() != (HERE / "controller-workflow.yml").read_bytes():
        raise ValueError("actual owned workflow differs from the validated standalone copy")
    recipe = config["on"]["workflow_dispatch"]["inputs"]["training_recipe"]
    if (recipe["type"] != "choice" or recipe["options"] != ["json-original-v1", "format-transfer-v1"]
            or recipe["default"] != "json-original-v1" or config["env"]["TRAINING_RECIPE"] != "${{ inputs.training_recipe }}"):
        raise ValueError("training recipe must use the closed bounded source choices")
    if config["permissions"] != {"contents": "read"} or set(config["on"]) != {"workflow_dispatch"}:
        raise ValueError("workflow must be manual and read-only")
    if set(config["jobs"]) != {"train", "audit", "serve"} or config["jobs"]["audit"]["needs"] != "train":
        raise ValueError("frozen independent audit ordering")
    if config["concurrency"]["cancel-in-progress"] != "false":
        raise ValueError("must preserve live work")
    for name, job in config["jobs"].items():
        if job["runs-on"] != "ubuntu-24.04" or not 1 <= int(job["timeout-minutes"]) <= 350:
            raise ValueError("only bounded standard public CPU jobs")
        for step in job["steps"]:
            if "uses" in step and not re.fullmatch(r"actions/[a-z-]+@[a-f0-9]{40}", step["uses"]):
                raise ValueError("unpinned action")
    if len(PINS["files"]) != 4 or any(not re.fullmatch(r"[a-f0-9]{64}", s) for s in PINS["files"].values()):
        raise ValueError("incomplete source hashes")
    if int(config["jobs"]["train"]["timeout-minutes"]) + int(config["jobs"]["audit"]["timeout-minutes"]) > 330:
        raise ValueError("training and audit exceed fixed free CPU budget")
    print("OWNED_WORKFLOW_SCHEMA_PASS")


if __name__ == "__main__":
    validate()
