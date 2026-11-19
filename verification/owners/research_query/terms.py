# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Retain upstream grants in successful and failed checkpoint bytes."""
import json
from pathlib import Path

from common import HERE, SHARED, module
from source import sha, write_json


def metadata():
    shared = SHARED / "qa-experiment"
    contract = json.loads((HERE / "contract.json").read_text())
    values = {}
    for name, key in (("LICENSE", "x-own-sand"), ("KIT-LICENSE", "x-upstream-mit"),
                      ("UPSTREAM-APACHE-2.0.txt", "x-upstream-apache-2.0")):
        raw = (shared / name).read_bytes()
        if sha(raw) != contract["licence_sha256"][name]:
            raise ValueError("reviewed licence changed")
        values[key] = raw.decode()
    rows = []
    for name, pin in contract["splits"].items():
        raw = (HERE / "data" / (name + ".jsonl")).read_bytes()
        if sha(raw) != pin["sha256"]:
            raise ValueError("attribution dataset identity changed")
        rows.extend(json.loads(line) for line in raw.splitlines())
    originals = {r["source_doi"]: {k: r[k] for k in ("source_url", "source_doi", "source_version", "source_sha256",
                 "source_license", "license_uri", "license_text", "copyright_statement", "authors", "article_title", "modification")}
                 for r in rows}
    values["x-source-attribution"] = json.dumps(list(originals.values()), ensure_ascii=True, separators=(",", ":"))
    values["x-upstream-notice"] = (HERE / "NOTICE").read_text()
    values["x-licence"] = "LicenseRef-zkasuran-SAND-1.0"
    return values


def stamp(path):
    contract = json.loads((HERE / "contract.json").read_text())
    source = SHARED / "qa-experiment/licensing.py"
    if sha(source.read_bytes()) != contract["shared_source_sha256"]["licensing.py"]:
        raise ValueError("reviewed inert stamp implementation changed")
    implementation = module("research_inert_licensing", source)
    # Configure an isolated imported module instance. No shared file is edited.
    implementation.licence_metadata = metadata
    implementation.preserve_artifact(path)


def preserve(root):
    root.mkdir(parents=True, exist_ok=True)
    values = metadata()
    for key, name in (("x-own-sand", "LICENSE"), ("x-upstream-mit", "KIT-LICENSE"),
                      ("x-upstream-apache-2.0", "UPSTREAM-APACHE-2.0.txt"), ("x-upstream-notice", "NOTICE")):
        (root / name).write_text(values[key])
    write_json(root / "source-attribution.json", json.loads(values["x-source-attribution"]))
    checks = []
    for directory in (root, root / "checkpoint", root / "initial", root / "adapter"):
        if directory.exists():
            for path in sorted(directory.iterdir()):
                if path.suffix in (".safetensors", ".pt") and path.is_file():
                    stamp(path)
                    checks.append({"file": str(path.relative_to(root)), "sha256": sha(path.read_bytes())})
    write_json(root / "licence-preservation.json", {"schema_version": 1, "checkpoints": checks,
               "own_additions": values["x-licence"], "upstream_grants_unchanged": True, "original_articles": len(json.loads(values["x-source-attribution"]))})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    preserve(parser.parse_args().root)
