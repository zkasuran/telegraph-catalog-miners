# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Complete licensed source capture and immutable post-freeze audit eligibility."""
import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import re
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import snapshot
import contracts

REPO = "jacobyoby/honeypot-blocklist"
FORMATS = ("blocklist.json", "blocklist.csv", "blocklist.misp.csv", "blocklist.txt")


@contextmanager
def baseline_files(root):
    if (root / "blocklist.json").is_file():
        yield root
        return
    archive = root / "original-source.zip"
    manifest = snapshot.decode(snapshot.read(root / "source-manifest.json"))
    if manifest.get("grant") != "CC0-1.0" or set(manifest.get("files", {})) != set(FORMATS):
        raise ValueError("original baseline source manifest changed")
    if archive.stat().st_size > 8 << 20:
        raise ValueError("baseline source archive ceiling")
    with tempfile.TemporaryDirectory(prefix="audit-baseline-", dir=root.parent) as scratch, zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(names) != len(FORMATS) or set(names) != set(FORMATS):
            raise ValueError("baseline archive has duplicate or unsafe members")
        for name in FORMATS:
            with source.open(name) as stream:
                raw = stream.read(snapshot.MAX_BYTES + 1)
            expected = manifest["files"][name]
            if len(raw) > snapshot.MAX_BYTES or len(raw) != expected["bytes"] or contracts.sha(raw) != expected["sha256"]:
                raise ValueError("original baseline source bytes changed")
            (Path(scratch) / name).write_bytes(raw)
        yield Path(scratch)


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Threat-IP-frozen-audit-source/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(snapshot.MAX_BYTES + 1)
        if response.status != 200 or urllib.parse.urlsplit(response.url).netloc != urllib.parse.urlsplit(url).netloc or len(raw) > snapshot.MAX_BYTES:
            raise ValueError("source origin/status/byte bound failed")
    return raw


def capture(commit, root):
    if not re.fullmatch(r"[a-f0-9]{40}", commit) or root.exists():
        raise ValueError("source capture needs immutable commit and fresh output")
    root.mkdir(parents=True)
    tree_raw = get(f"https://api.github.com/repos/{REPO}/git/trees/{commit}?recursive=1")
    tree = snapshot.decode(tree_raw)
    if tree.get("truncated") is not False or tree.get("sha") != commit:
        raise ValueError("complete source tree identity required")
    (root / "tree.json").write_bytes(tree_raw)
    required = set(FORMATS) | {"LICENSE", "README.md", "validate.py", "go.mod"}
    selected = [entry for entry in tree["tree"] if entry["type"] == "blob" and
                (entry["path"] in required or entry["path"].startswith("internal/validator/") or entry["path"].startswith("cmd/blocklist-validator/"))]
    if not required <= {entry["path"] for entry in selected}:
        raise ValueError("source publication/validator inventory incomplete")
    files = {}
    total = len(tree_raw)
    for entry in selected:
        name = entry["path"]
        if ".." in Path(name).parts or Path(name).is_absolute():
            raise ValueError("source path escape")
        raw = get(f"https://raw.githubusercontent.com/{REPO}/{commit}/{name}")
        total += len(raw)
        if total > 8 << 20 or hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != entry["sha"]:
            raise ValueError("complete source bytes/tree binding failed")
        target = root / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        files[name] = {"sha256": contracts.sha(raw), "bytes": len(raw)}
    licence = (root / "source/LICENSE").read_bytes()
    if b"waived all copyright and" not in licence or b"this blocklist and its generator" not in licence or b"CC0" not in licence:
        raise ValueError("explicit original data/generator grant not verified")
    receipt = {"schema_version": 1, "source_repo": REPO, "source_commit": commit, "files": files,
               "source_tree_entries": len(tree["tree"]), "complete_inventory": True, "grants_verified": True,
               "original_grant": "CC0-1.0", "total_bytes": total, "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "boundary": "new audit source only, never model fitting or selection; no protected/exposed evaluation pools"}
    contracts.write_json(root / "receipt.json", receipt)
    return receipt


def eligible(old_rows, new_rows, development_ips, frozen_at):
    frozen = dt.datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
    # Old/new versions of the same IP bridge changed ASNs through their /24.
    # Closure covers every retained source row, not only selected development IPs.
    components = snapshot.groups(old_rows + new_rows)
    new_by_ip = {row["ip"]: row for row in new_rows}
    selected = []
    excluded = {"development_overlap": 0, "unknown_asn": 0, "no_post_freeze_observation": 0}
    for group in components:
        if any(row["ip"] in development_ips for row in group["rows"]):
            excluded["development_overlap"] += 1
            continue
        if group["unknown_asn"]:
            excluded["unknown_asn"] += 1
            continue
        fresh = {row["ip"]: new_by_ip[row["ip"]] for row in group["rows"] if row["ip"] in new_by_ip
                 and snapshot.instant(new_by_ip[row["ip"]]["last_seen"]) > frozen}
        if not fresh:
            excluded["no_post_freeze_observation"] += 1
            continue
        selected.append({"group_id": group["group_id"], "rows": list(fresh.values())})
    return selected, excluded, len(components)


def preflight(lock, capture_root, base_source, development):
    if lock["audit_used_for_selection"] is not False:
        raise ValueError("candidate is not frozen independently of audit")
    with baseline_files(base_source) as baseline:
        old_meta, old_rows, old_curation = snapshot.curate_formats(baseline)
    meta, rows, curation = snapshot.curate_formats(capture_root / "source")
    if meta["window_days"] != old_meta["window_days"]:
        raise ValueError("audit source recency contract changed")
    development_ips = {row["source_ip"] for row in development}
    selected, excluded, total = eligible(old_rows, rows, development_ips, lock["frozen_at"])
    report = {"schema_version": 1, "family": contracts.FAMILY, "status": "AUDIT_SOURCE_READY" if len(selected) >= 32 else "AUDIT_SOURCE_NOT_READY",
              "package_sha256": lock["package_sha256"], "candidate_frozen_at": lock["frozen_at"],
              "source_updated": meta["updated"], "source_rows": len(rows), "raw_curation": curation,
              "global_components": total, "eligible_independent_groups": len(selected), "minimum_groups": 32,
              "excluded_components": excluded, "audit_used_for_selection": False, "local_model_operations": 0,
              "policy": "strict post-freeze last_seen and global old/new /24+ASN closure, no changed source values or lowered gates"}
    contracts.write_json(capture_root / "source-preflight.json", report)
    return meta, selected, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--development", type=Path, nargs=2, required=True)
    args = parser.parse_args()
    lock = snapshot.decode(args.lock.read_bytes())
    if {contracts.sha(path.read_bytes()) for path in args.development} != {lock["train_sha256"], lock["validation_sha256"]}:
        raise ValueError("complete frozen train/validation entity inventories required")
    development = [json.loads(line) for path in args.development for line in path.read_bytes().splitlines()]
    capture(args.commit, args.out)
    _, _, report = preflight(lock, args.out, args.base_source, development)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
