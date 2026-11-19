# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Run the real compiler/interpreter harness against incorrect label and syntax."""
import argparse
import json
import subprocess
from pathlib import Path

from data import sha, write
from policy import execution_gate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    root = p.parse_args().root
    raw = (root / "data" / "inventory.json").read_bytes()
    rows = json.loads(raw)
    report = json.loads((root / "evm-report.json").read_text())
    execution_gate(rows, report, sha(raw))
    failures = []
    for name in ("flipped_label", "invalid_solidity"):
        row = dict(rows[0])
        if name == "flipped_label":
            row["vulnerable"] = not row["vulnerable"]
        else:
            row["source"] += " THIS IS INVALID SOLIDITY"
            row["source_sha256"] = sha(row["source"].encode())
        path, evidence = root / (name + ".json"), root / (name + ".log")
        write(path, [row])
        result = subprocess.run(["node", str(Path(__file__).parent / "execute.mjs"), str(path), str(root / (name + "-result.json"))],
                                capture_output=True, text=True, timeout=30)
        evidence.write_text(result.stdout + result.stderr)
        if result.returncode == 0:
            raise ValueError("primary interpreter did not catch " + name)
        failures.append({"control": name, "exit_code": result.returncode, "log": evidence.name})
    write(root / "failure-controls.json", {"positive_execution_binding": "pass", "controls": failures})
    print(json.dumps({"positive_execution_binding": "pass", "controls": failures}))


if __name__ == "__main__":
    main()
