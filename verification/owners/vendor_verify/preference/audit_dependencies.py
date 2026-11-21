# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Audit every locked release and fail on skipped dependencies."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def requirements(text):
    packages = re.findall(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)", text, re.MULTILINE)
    if not packages or len(packages) != len({name.lower() for name, _ in packages}):
        raise ValueError("empty or duplicate locked dependency list")
    rows, mapped = [], {}
    for name, version in packages:
        if name == "torch" and version == "2.15.0+cpu":
            mapped[name] = {
                "installed_variant": version,
                "advisory_release": "2.15.0",
                "scope": "Release-level PyPI advisories; CPU wheel bytes are separately hash-pinned, not vulnerability-scanned.",
            }
            version = "2.15.0"
        elif "+" in version:
            raise ValueError("unmapped non-PyPI local dependency version")
        rows.append(f"{name}=={version}\n")
    return "".join(rows), mapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    text, mapped = requirements(args.lock.read_text())
    args.out.write_text(text)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip_audit",
            "--disable-pip",
            "--no-deps",
            "--progress-spinner",
            "off",
            "-f",
            "json",
            "-r",
            str(args.out),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if not result.stdout.strip():
        raise RuntimeError("auditor produced no report: " + result.stderr[:500])
    report = json.loads(result.stdout)
    dependencies = report["dependencies"]
    skipped = [d for d in dependencies if d.get("skip_reason")]
    vulnerabilities = sum(len(d.get("vulns", [])) for d in dependencies)
    final = {
        "mapped_variants": mapped,
        "dependencies": dependencies,
        "skipped": len(skipped),
        "vulnerabilities": vulnerabilities,
        "verdict": "PASS"
        if result.returncode == 0 and not skipped and not vulnerabilities
        else "FAIL",
    }
    args.report.write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps({k: v for k, v in final.items() if k != "dependencies"}))
    return 0 if final["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
