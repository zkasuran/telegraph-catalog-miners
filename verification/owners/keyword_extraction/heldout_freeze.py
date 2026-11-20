# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Immutable validation/audit source bytes before any model selection or audit."""
import argparse
import json
from pathlib import Path

from source import sha, write


def wire(rows):
    return "".join(json.dumps({k: v for k, v in r.items() if k != "provisional_split"}, ensure_ascii=False, allow_nan=False) + "\n" for r in rows).encode()


def verify_freeze(splits, manifest):
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1 or manifest.get("family") != "KEYWORD_EXTRACTION" or manifest.get("status") != "HELDOUT_SOURCE_FROZEN":
        raise ValueError("exact held-out source freeze required")
    if set(manifest.get("files", {})) != {"validation", "test"}:
        raise ValueError("both frozen held-out source files required")
    for name, pairs, groups in (("validation", 16, 8), ("test", 64, 32)):
        rows = splits[name]
        actual = {"sha256": sha(wire(rows)), "pairs": len(rows), "groups": len({r["group_id"] for r in rows})}
        if actual != manifest["files"][name] or actual["pairs"] < pairs or actual["groups"] < groups:
            raise ValueError("frozen held-out source bytes/counts changed: " + name)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    path = args.reviewed / "reviewed-pairs.jsonl"
    if path.stat().st_size > 8 << 20:
        raise ValueError("reviewed source byte ceiling exceeded")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    splits = {s: [r for r in rows if r["provisional_split"] == s] for s in ("validation", "test")}
    manifest = {"schema_version": 1, "family": "KEYWORD_EXTRACTION", "status": "HELDOUT_SOURCE_FROZEN",
                "source_manifest_sha256": sha((args.reviewed / "reviewed-manifest.json").read_bytes()),
                "files": {name: {"sha256": sha(wire(data)), "pairs": len(data), "groups": len({r["group_id"] for r in data})} for name, data in splits.items()},
                "scope": "Explicitly AI-assisted source preferences. Frozen source bytes, not measured model quality or independent human review."}
    verify_freeze(splits, manifest)
    args.out.mkdir(exist_ok=True)
    target = args.out / "heldout-freeze.json"
    if target.exists():
        existing = json.loads(target.read_text())
        verify_freeze(splits, existing)
        print(json.dumps(existing))
        return
    for name, data in splits.items():
        (args.out / (name + ".jsonl")).write_bytes(wire(data))
    write(target, manifest)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
