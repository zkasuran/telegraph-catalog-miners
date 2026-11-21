# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Add unused source groups to training without moving any reserved split."""

from pathlib import Path

import data

EXTRA_TRAIN_GROUPS = 512


def extend(pool, original, extra=EXTRA_TRAIN_GROUPS):
    if type(extra) is not int or not 1 <= extra <= 512:
        raise ValueError("bounded predeclared training extension required")
    data.check_splits(original)
    reserved = {row["group_id"] for rows in original.values() for row in rows}
    unused = [row for row in pool if row["group_id"] not in reserved]
    if len(unused) < extra:
        raise ValueError("insufficient unused source groups, no padding")
    result = {**original, "train": original["train"] + unused[:extra]}
    data.check_splits(result)
    for split in ("validation", "audit"):
        if result[split] is not original[split]:
            raise ValueError("reserved split moved")
    return result


def prepare(source, expected, out):
    out = Path(out)
    if out.exists():
        raise ValueError("expanded source checkpoint already retained")
    pairs, counts = data.read_source(source)
    pool, excluded = data.grouped(pairs)
    original = data.allocate(pool)
    encoded = {
        split: b"".join(data.encode(row) for row in rows)
        for split, rows in original.items()
    }
    actual = {
        split: {
            "sha256": data.sha(raw),
            "bytes": len(raw),
            "pairs": len(original[split]),
        }
        for split, raw in encoded.items()
    }
    if actual != expected["files"]:
        raise ValueError("original split identity drift")
    expanded = extend(pool, original)
    train_raw = b"".join(data.encode(row) for row in expanded["train"])
    report = {
        "schema_version": 1,
        "family": "VENDOR_VERIFY",
        "source_sha256": data.SOURCE_SHA256,
        "source_counts": counts,
        "eligible_independent_components": len(pool),
        "exclusions": excluded,
        "original_files": actual,
        "training": {
            "pairs": len(expanded["train"]),
            "sha256": data.sha(train_raw),
            "bytes": len(train_raw),
        },
        "extra_train_groups": EXTRA_TRAIN_GROUPS,
        "normalized_overlap": 0,
        "allocation": "original allocation preserved, next 512 unused salted-order groups added to training only",
        "audit_evaluated": False,
        "fit_performed": False,
        "registration_ready": False,
    }
    out.mkdir(mode=0o700, parents=True)
    (out / "train.jsonl").write_bytes(train_raw)
    (out / "EXPANSION-REPORT.json").write_bytes(data.encode(report))
    return expanded["train"], train_raw, report
