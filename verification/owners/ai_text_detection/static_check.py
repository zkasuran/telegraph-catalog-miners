# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Offline package, syntax and workflow validation. No neural imports."""
import ast
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def validate_checkout(config):
    expected = {"/verification/owners/ai_text_detection/", "/.github/workflows/evaluator-owner-ai_text_detection.yml"}
    for job in config["jobs"].values():
        checkout = job["steps"][0]
        settings = checkout.get("with", {})
        if (not checkout.get("uses", "").startswith("actions/checkout@") or
                settings.get("persist-credentials") is not False or
                settings.get("sparse-checkout-cone-mode") is not False or
                set(settings.get("sparse-checkout", "").split("\n")) != expected):
            raise ValueError("standalone checkout must include the exact source and workflow")


def main():
    for path in HERE.glob("*.py"):
        compile(path.read_text(), str(path), "exec")
        ast.parse(path.read_text(), filename=str(path), feature_version=(3, 11))
    for path in HERE.glob("*.json"):
        json.loads(path.read_bytes())
    for name in ("LICENSE", "NOTICE", "DATA-SOURCES.md", "SECURITY.md", "README.md", "requirements.lock"):
        if not (HERE / name).is_file() or not (HERE / name).stat().st_size:
            raise ValueError("missing standalone release asset: " + name)
    workflow = json.loads((HERE / "workflow.json").read_text())
    validate_checkout(workflow)
    path = HERE.parents[2] / ".github/workflows/evaluator-owner-ai_text_detection.yml"
    if path.read_bytes() != (HERE / "workflow.json").read_bytes():
        raise ValueError("owned workflow differs from validated JSON/YAML config")
    if set(workflow["jobs"]) != {"verify", "serve"} or workflow["permissions"] != {"contents": "read"}:
        raise ValueError("workflow job or permission schema mismatch")
    for job in workflow["jobs"].values():
        if job["runs-on"] != "ubuntu-24.04" or not 1 <= job["timeout-minutes"] <= 350:
            raise ValueError("unbounded or nonpublic CPU job")
        if "strategy" in job:
            raise ValueError("unexpected job fanout")
        for step in job["steps"]:
            if "uses" in step and not re.fullmatch(r"[A-Za-z0-9_/-]+@[0-9a-f]{40}", step["uses"]):
                raise ValueError("unpinned workflow action")
    lock = (HERE / "requirements.lock").read_text()
    if "--hash=sha256:" not in lock or "torch==2.15.0+cpu" not in lock:
        raise ValueError("hash-locked CPU dependencies missing")
    # A source gate must not import a neural package directly or indirectly.
    local = ("common.py", "source.py", "wire.py", "checkpoint_contract.py", "development_contract.py", "reference_data_split.py", "preserve.py", "test_source.py", "static_check.py")
    for name in local:
        for node in ast.walk(ast.parse((HERE / name).read_text())):
            modules = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            if any(m and m.split(".")[0] in {"torch", "transformers", "peft", "numpy"} for m in modules):
                raise ValueError("neural import in local release path")
    recipe = json.loads((HERE / "recipe.json").read_text())
    if (recipe["schema_version"] != 1 or recipe["epochs"] > 8 or recipe["grad_accum"] != 8 or
            recipe["lora_rank"] != 8 or recipe["lora_alpha"] != 32 or recipe["original_run_id"] != 44565365361):
        raise ValueError("unbounded or unbound continuation recipe")
    print("PASS: compiled Python syntax, JSON configs, bounded pinned public workflow and local import boundary")


if __name__ == "__main__":
    main()
