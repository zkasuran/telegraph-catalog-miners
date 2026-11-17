# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Prove envelope regressions fail against deliberately weakened source copies."""
import json
import shutil
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / "worker.mjs").read_text()
mutations = [
    ("remove-exact-id", "data.id !== 1", "false"),
    ("allow-unknown-marker", "const legacy = !Object.hasOwn(data, 'jsonrpc');", "const legacy = true;"),
    ("allow-ambiguous-legacy-error", "(legacy && data.result !== null)", "false"),
]
reports = []
for name, before, after in mutations:
    assert source.count(before) == 1, name
    directory = root / ".envelope-mutations" / name
    directory.mkdir(parents=True, exist_ok=True)
    for filename in ["core.mjs", "proof.mjs", "test-envelope.mjs"]:
        shutil.copyfile(root / filename, directory / filename)
    (directory / "worker.mjs").write_text(source.replace(before, after, 1))
    (directory / "evidence").mkdir(exist_ok=True)
    shutil.copyfile(root / "evidence/fixtures-public.json", directory / "evidence/fixtures-public.json")
    result = subprocess.run(["node", "--test", "test-envelope.mjs"], cwd=directory, capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode != 0 and "not ok" in result.stdout, name
    reports.append({"mutation": name, "returncode": result.returncode, "caught": True,
                    "failed_tests": [line for line in result.stdout.splitlines() if line.startswith("not ok")],
                    "stdout": result.stdout, "stderr": result.stderr})
    print("MUTATION CAUGHT", name)
(root / "evidence/envelope-mutations.json").write_text(json.dumps({"all_caught": True, "mutations": reports}, indent=2) + "\n")
(root / "evidence/envelope-mutation-summary.json").write_text(json.dumps({"all_caught": True, "mutations": [{key: value for key, value in record.items() if key not in {"stdout", "stderr"}} for record in reports]}, indent=2) + "\n")
