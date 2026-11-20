# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only verification of diagnostic selection and uncropped control text."""
import json

from data import HERE
from diagnostic import training_controls


def main():
    manifest = json.loads((HERE / "data/manifest.json").read_text())
    train = [row for g in manifest["splits"]["train"] for row in g["pairs"]]
    controls = training_controls(manifest, train)
    docs = {d["doi"]: d for g in manifest["splits"]["train"] for d in g["documents"]}
    forbidden = {d["doi"] for name in ("validation", "audit") for g in manifest["splits"][name] for d in g["documents"]}
    for name, rows in controls.items():
        if len(rows) != 8 or len({r["group_id"] for r in rows}) != 4 or {r["source_id"] for r in rows} & forbidden:
            raise ValueError("diagnostic controls must be four training-only groups")
        for row in rows:
            if row["chosen"] != docs[row["source_id"]]["title"] or row["rejected"] != docs[row["donor_id"]]["title"]:
                raise ValueError("diagnostic original title changed")
            if name == "single_requested_reference_control" and not row["prompt"].endswith(docs[row["source_id"]]["abstract"]):
                raise ValueError("diagnostic complete requested source was cropped")
    return {"diagnostic_source": "PASS", "training_groups": 4, "validation_used": False, "audit_used": False}


if __name__ == "__main__":
    print(json.dumps(main()))
