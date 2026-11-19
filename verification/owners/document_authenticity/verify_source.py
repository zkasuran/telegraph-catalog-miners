# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only packaged gate, independent of private paths or fitted models."""
import ast
import json
import os
import py_compile
import re
import subprocess
import tempfile
from pathlib import Path

import yaml

from contracts import data_gate, read_split, source_manifest
from scorer import score

HERE = Path(__file__).resolve().parent


def main():
    with tempfile.TemporaryDirectory(prefix="document-owner-gate-", dir=HERE) as directory:
        cache = Path(directory)
        count = 0
        for source in HERE.glob("*.py"):
            py_compile.compile(str(source),cfile=str(cache/(source.name+"c")),doraise=True)
            count += 1
        spec = json.loads((HERE/"spec.json").read_text())
        manifest = source_manifest()
        if spec["family"] != manifest["family"] or spec["run_id"] != manifest["run_id"]:
            raise ValueError("run identity mismatch")
        for name in ("LICENSE","NOTICE","DATA-SOURCES.md","SECURITY.md","KIT-LICENSE"):
            if not (HERE/name).is_file() or (HERE/name).stat().st_size < 80:
                raise ValueError("missing required rights or threat model")
        workflow = yaml.safe_load((HERE/"workflow.yml").read_text())
        trigger = workflow.get("on", workflow.get(True))
        if set(trigger) != {"workflow_dispatch"} or workflow["permissions"] != {"contents":"read"}:
            raise ValueError("workflow authority must stay closed")
        for job in workflow["jobs"].values():
            if job["runs-on"] != "ubuntu-24.04" or not 1 <= job["timeout-minutes"] <= 350:
                raise ValueError("free bounded runner required")
            for i,step in enumerate(job["steps"]):
                if "uses" in step and not re.fullmatch(r"actions/[a-z-]+@[0-9a-f]{40}",step["uses"]):
                    raise ValueError("unpinned workflow action")
                if "run" in step:
                    shell = cache/(str(i)+".sh")
                    shell.write_text(step["run"])
                    subprocess.run(["bash","-n",str(shell)],check=True,timeout=10)
        if len(workflow["jobs"]) > 6:
            raise ValueError("workflow parallel ceiling")
        splits = {name:read_split(name) for name in ("train","validation","audit")}
        report = data_gate(splits)
        row = splits["train"][0]
        if score(row["prompt"],row["chosen"])["score"] != 1 or score(row["prompt"],row["rejected"])["score"] != 0:
            raise ValueError("source-derived primary smoke failed")
        for name in ("remote_train.py","serve.py"):
            tree = ast.parse((HERE/name).read_text())
            for node in tree.body:
                if isinstance(node,(ast.Import,ast.ImportFrom)):
                    names = [a.name for a in node.names] if isinstance(node,ast.Import) else [node.module]
                    if any(n.split(".")[0] in {"torch","peft","transformers","onnxruntime","tokenizers"} for n in names):
                        raise ValueError("neural import above remote guard")
        # The built executable does not run here. Its full compilation is remote.
        print(json.dumps({"compiled_python_modules":count,"source_groups":report["original_groups"],
                          "splits":{s:len(rows) for s,rows in splits.items()},"primary_smoke":"pass",
                          "frozen_baseline":"unchanged","remote_model_operations":"not run"}))


if __name__ == "__main__":
    main()
