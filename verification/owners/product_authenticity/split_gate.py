# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Refuse a fabricated train/validation/audit split of one connected study."""
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parent


def identity(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def check_split(partitions):
    if set(partitions) != {"train", "validation", "audit"}:
        raise ValueError("all three partitions required")
    seen = {}
    for partition, rows in partitions.items():
        if not rows:
            raise ValueError("empty partition: " + partition)
        for row in rows:
            keys = [
                "source:" + identity(row["source_doi"].removesuffix(".st001")),
                "specimen:" + identity(row["record_id"]),
                "event:" + identity(row["acquisition_event"]),
                "entity:" + identity(row["entity_group"]),
                "batch:" + identity(row["batch_dates"].split()[0]),
            ]
            if row["counterfeit_family_group"]:
                keys.append("family:" + identity(row["counterfeit_family_group"]))
            for key in keys:
                if key in seen and seen[key] != partition:
                    raise ValueError("group leakage: " + key)
                seen[key] = partition
    return True


def main():
    rows = json.loads((ROOT / "SOURCE-INVENTORY.json").read_text())["records"]
    # A tempting time split is intentionally shown to fail, not used for training.
    partitions = {"train": rows[:5], "validation": rows[5:23], "audit": rows[23:]}
    try:
        check_split(partitions)
    except ValueError as exc:
        report = {"passed": False, "model_dispatch_allowed": False, "reason": str(exc),
                  "source_studies": 1, "record_units": len(rows), "source_held_out_score": None,
                  "actual_split": "all records are diagnostic fixtures only",
                  "next_data": "At least two more independent rights-cleared source studies with item-level labels on both sides, disjoint brands/entities and acquisition events. Verify overlapping specimens and counterfeit families before allocating whole groups."}
        (ROOT / "SPLIT-GATE.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, sort_keys=True))
        return 2
    raise RuntimeError("expected the single-source split to be rejected")


if __name__ == "__main__":
    raise SystemExit(main())
