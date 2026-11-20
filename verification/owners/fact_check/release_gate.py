# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Standalone source gate. No model imports, downloads or audit score selection."""
import ast
import json
import shutil

from data import COUNTS, HERE, digest, read_splits, render
from licensing import licence_metadata
from source_pack import unpack


def restore_policy():
    original = HERE / "CC-BY-SA-4.0-policy.html"
    expected = json.loads((HERE / "sources.json").read_text())["grants"]["source-policy.html"]
    if digest(original.read_bytes()) != expected:
        raise ValueError("original policy grant hash mismatch")
    target = HERE / "source-policy.html"
    if target.is_symlink() or target.exists() and target.read_bytes() != original.read_bytes():
        raise ValueError("existing policy cache changed")
    if not target.exists():
        shutil.copyfile(original, target)


def main():
    restore_policy()
    unpack()
    contract = json.loads((HERE / "contract.json").read_text())
    for name, expected in contract["freeze_sha256"].items():
        path = HERE / name if name == "sources.json" else HERE / "bundle" / name
        if digest(path.read_bytes()) != expected:
            raise ValueError("frozen artifact changed: " + name)
    data, splits = read_splits(HERE / "bundle")
    if data["training_allowed"] is not True or {name: len(rows) for name, rows in splits.items()} != COUNTS:
        raise ValueError("independent source pilot incomplete")
    visible = json.loads((HERE / "bundle/token-visible-binding.json").read_text())
    if visible["max_tokens"] != 256 or visible["cropped"] is not False:
        raise ValueError("source visibility boundary changed")
    all_text, all_groups, maximum = set(), set(), 0
    for name, rows in splits.items():
        groups = {r["group_id"] for r in rows}
        texts = {render(r["prompt"], r[side]) for r in rows for side in ("chosen", "rejected")}
        if groups & all_groups or texts & all_text:
            raise ValueError("independent source overlap")
        all_groups.update(groups)
        all_text.update(texts)
        records = {record["group_id"]: record for record in visible["splits"][name]}
        for row in rows:
            for side in ("chosen", "rejected"):
                record = records[row["group_id"]][side]
                if record["render_sha256"] != digest(render(row["prompt"], row[side]).encode()) or not 1 <= record["tokens"] <= 256:
                    raise ValueError("frozen visible input binding changed")
                maximum = max(maximum, record["tokens"])
    licence_metadata()
    for name in ("run.py", "remote.py", "report.py", "serving.py"):
        ast.parse((HERE / name).read_text())
    from static_check import main as static_check
    static_check()
    print(json.dumps({"status": "SOURCE_GATE_PASS", "groups": len(all_groups), "maximum_recorded_tokens": maximum,
                      "source_manifest_sha256": digest((HERE / "sources.json").read_bytes()),
                      "baseline_refitted": False, "audit_scored": False, "neural_imports": False}))


if __name__ == "__main__":
    main()
