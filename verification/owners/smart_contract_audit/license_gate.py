# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bind original grants, exact dependency bytes and accepted source labels."""
import argparse
import json
from pathlib import Path

from data import sha, write
from policy import execution_gate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    root = p.parse_args().root
    here = Path(__file__).resolve().parent
    licence = json.loads((here / "licences.json").read_text())
    assert (here / "LICENSE").read_text() == licence["sand"]
    assert "Apache License" in licence["apache_2_0"] and "Copyright (c) 2026 telegraphprotocol" in licence["kit_mit"]
    assert "Source Code Form" in (here / "MPL-2.0.txt").read_text()
    assert licence["torch_bsd"] == (here / "PYTORCH-LICENSE.txt").read_text()
    assert "Redistribution and use in source and binary forms" in licence["torch_bsd"]
    raw = (root / "data/inventory.json").read_bytes()
    rows = json.loads(raw)
    execution_gate(rows, json.loads((root / "evm-report.json").read_text()), sha(raw))
    assert all(r["source"].startswith("// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0\n") for r in rows)
    assert all(r["authorship"].startswith("Original AI-assisted") and r["licence"] == "LicenseRef-zkasuran-SAND-1.0" for r in rows)
    assert not any("@vulnerable" in r["source"] for r in rows)
    report = {"grants_verified": True, "original_source_rows": len(rows), "imported_contracts": 0,
              "licences_sha256": sha((here / "licences.json").read_bytes()),
              "source_manifest_sha256": sha((here / "data-bindings.json").read_bytes()),
              "package_lock_sha256": sha((here / "package-lock.json").read_bytes()),
              "authorship": "Original AI-assisted source and constructed preferences; no human review"}
    write(root / "grant-report.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
