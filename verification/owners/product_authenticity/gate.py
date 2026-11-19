# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Local source-only gate and honest remote metadata report. No neural imports."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

from source_inventory import readiness

ROOT = Path(__file__).resolve().parent


def main():
    seal = json.loads((ROOT / "SOURCE-MANIFEST.json").read_text())
    for entry in seal["files"]:
        path = ROOT / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("frozen source mismatch: " + entry["path"])
    code = ("scorer.py", "inventory.py", "split_gate.py", "test_product.py", "source_inventory.py", "product_evidence.py", "test_expansion.py", "gate.py")
    for name in code:
        ast.parse((ROOT / name).read_text(), filename=name)
        compile((ROOT / name).read_text(), name, "exec")
    report = readiness()
    report.update(source_manifest_sha256=hashlib.sha256((ROOT / "SOURCE-MANIFEST.json").read_bytes()).hexdigest(),
                  commit=os.environ.get("GITHUB_SHA"), run_id=int(os.environ["GITHUB_RUN_ID"]) if os.environ.get("GITHUB_RUN_ID") else None,
                  runtime=sys.version.split()[0], status="SOURCE_DIAGNOSTIC_ONLY", all_supported_formats_tested=False,
                  registration_ready=False)
    directory = os.environ.get("PRODUCT_REPORT_DIR")
    if directory:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / "source-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
