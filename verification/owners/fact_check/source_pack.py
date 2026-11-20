# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Compact exact primary bytes for the patch. Source values are never edited."""
import argparse
import gzip
import io
import json
import tarfile
from pathlib import Path

from data import HERE, TOKENIZER_SHA, digest, write_json


def pack():
    root = HERE / "sources"
    index = json.loads((root / "index.json").read_text())
    if index.get("complete") is not True or (HERE / "contract.json").exists():
        raise ValueError("pack once after source completion and before split freeze")
    records = [record for record in index["records"]
               if record["file"] == "discovery.json" or "-info-n" in record["file"] or record["file"].startswith("objects-")]
    # Pre-revision transport attempts are retained locally but not training inputs.
    index["records"] = records
    write_json(root / "index.json", index)
    files = {record["file"]: (root / record["file"]).read_bytes() for record in records}
    if any(digest(files[r["file"]]) != r["sha256"] for r in records):
        raise ValueError("primary source hashes changed before packing")
    files["index.json"] = (root / "index.json").read_bytes()
    files["tokenizer.json"] = (root / "tokenizer.json").read_bytes()
    files["SOURCE-CC0.txt"] = (HERE / "SOURCE-CC0.txt").read_bytes()
    if digest(files["tokenizer.json"]) != TOKENIZER_SHA:
        raise ValueError("tokenizer bytes changed")
    with (HERE / "sources.tar.gz").open("wb") as output, gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed, tarfile.open(fileobj=compressed, mode="w") as archive:
        for name, raw in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(raw)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(raw))
    report = {"archive_sha256": digest((HERE / "sources.tar.gz").read_bytes()),
              "archive_bytes": (HERE / "sources.tar.gz").stat().st_size,
              "raw_bytes": sum(map(len, files.values())), "entries": len(files),
              "policy": "Exact primary response bytes and CC0; transport-only pre-revision captures excluded"}
    write_json(HERE / "source-archive.json", report)
    print(json.dumps(report, indent=2))


def unpack():
    spec = json.loads((HERE / "sources.json").read_text())
    path = HERE / "sources.tar.gz"
    if path.is_symlink() or path.stat().st_size > 32 << 20 or digest(path.read_bytes()) != spec["source_archive_sha256"]:
        raise ValueError("frozen source archive identity or ceiling")
    root = HERE / "sources"
    root.mkdir(exist_ok=True)
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(members) > 192 or len(set(names)) != len(names) or sum(m.size for m in members) > 96 << 20:
            raise ValueError("source archive expansion ceiling")
        for member in members:
            if not member.isfile() or Path(member.name).name != member.name or member.size > 8 << 20:
                raise ValueError("archive path/type/entry ceiling")
            target = root / member.name
            raw = archive.extractfile(member).read()
            if target.is_symlink() or (target.exists() and target.read_bytes() != raw):
                raise ValueError("existing frozen source differs")
            if not target.exists():
                target.write_bytes(raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unpack", action="store_true")
    args = parser.parse_args()
    unpack() if args.unpack else pack()
