# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Audit every locked release and fail on skipped dependencies."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from common import MAX_BODY, strict_json


def canonical_name(name):
    # Python Packaging name-normalization specification, PEP 503 normalized names.
    if (not isinstance(name, str) or len(name) > 128
            or re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", name) is None):
        raise ValueError("invalid dependency name")
    return re.sub(r"[-_.]+", "-", name).lower()


def requirements(text):
    if not isinstance(text, str) or len(text.encode()) > 2 << 20:
        raise ValueError("locked requirements exceed source ceiling")
    packages = []
    for line in text.splitlines():
        line = line.strip().removesuffix("\\").strip()
        if not line or line.startswith("#"):
            continue
        if re.fullmatch(r"--hash=sha256:[0-9a-f]{64}", line):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;\\]+)", line)
        if match is None:
            raise ValueError("unsupported or unpinned locked dependency row")
        name, version = match.groups()
        if len(version) > 128:
            raise ValueError("dependency version exceeds ceiling")
        packages.append((canonical_name(name), version))
    if not packages or len(packages) > 256 or len(packages) != len({name for name, _ in packages}):
        raise ValueError("empty or duplicate locked dependency list")
    rows, mapped = [], {}
    for name, version in packages:
        if name == "torch" and version == "2.15.0+cpu":
            mapped[name] = {"installed_variant": version, "advisory_release": "2.15.0",
                            "scope": "Release-level PyPI advisories; CPU wheel bytes are separately hash-pinned, not vulnerability-scanned."}
            version = "2.15.0"
        elif "+" in version:
            raise ValueError("unmapped non-PyPI local dependency version")
        rows.append(f"{name}=={version}\n")
    return "".join(rows), mapped


def validate_report(report, audited, returncode):
    expected = dict(line.strip().split("==", 1) for line in audited.splitlines())
    if (not isinstance(report, dict) or set(report) != {"dependencies", "fixes"}
            or not isinstance(report["dependencies"], list) or len(report["dependencies"]) > 256
            or report["fixes"] != []):
        raise ValueError("auditor JSON schema differs from the pinned no-fix contract")
    dependencies = report["dependencies"]
    names, skipped, vulnerabilities = set(), 0, 0
    for dep in dependencies:
        if not isinstance(dep, dict):
            raise ValueError("malformed auditor dependency")
        name = canonical_name(dep.get("name"))
        if name not in expected or name in names:
            raise ValueError("duplicate or unexpected auditor dependency")
        names.add(name)
        if "skip_reason" in dep:
            if set(dep) != {"name", "skip_reason"} or not isinstance(dep["skip_reason"], str) or not dep["skip_reason"]:
                raise ValueError("malformed auditor skip record")
            skipped += 1
            continue
        if set(dep) != {"name", "version", "vulns"} or dep["version"] != expected[name] or not isinstance(dep["vulns"], list):
            raise ValueError("auditor release or vulnerability schema differs from the lock")
        for vuln in dep["vulns"]:
            if (not isinstance(vuln, dict) or not isinstance(vuln.get("id"), str) or not vuln["id"]
                    or not isinstance(vuln.get("fix_versions"), list)
                    or any(not isinstance(version, str) for version in vuln["fix_versions"])):
                raise ValueError("malformed auditor vulnerability record")
        vulnerabilities += len(dep["vulns"])
    if names != set(expected):
        raise ValueError("auditor omitted locked dependencies")
    complete = not skipped
    return {"dependencies": dependencies, "expected_dependencies": len(expected), "reported_dependencies": len(dependencies),
            "coverage_verified": complete, "skipped": skipped, "vulnerabilities": vulnerabilities,
            "verdict": "PASS" if type(returncode) is int and returncode == 0 and complete and not vulnerabilities else "FAIL"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    text, mapped = requirements(args.lock.read_text())
    args.out.write_text(text)
    result = subprocess.run([sys.executable, "-m", "pip_audit", "--disable-pip", "--no-deps",
                             "--progress-spinner", "off", "-f", "json", "-r", str(args.out)],
                            capture_output=True, text=True, timeout=180, check=False)
    try:
        if not result.stdout.strip() or len(result.stdout.encode()) > MAX_BODY:
            raise ValueError("auditor report absent or exceeds byte ceiling")
        final = {"mapped_variants": mapped, **validate_report(strict_json(result.stdout), text, result.returncode)}
    except (ValueError, TypeError, RecursionError):
        # Unknown coverage is not zero vulnerabilities and never earns a pass.
        final = {"mapped_variants": mapped, "dependencies": [], "coverage_verified": False,
                 "expected_dependencies": len(text.splitlines()), "reported_dependencies": None,
                 "skipped": None, "vulnerabilities": None, "verdict": "FAIL",
                 "error": "Auditor report is malformed, incomplete or not bound to every locked release."}
    args.report.write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps({k: v for k, v in final.items() if k != "dependencies"}))
    return 0 if final["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
