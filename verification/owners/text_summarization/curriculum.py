# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Train-only complete-source warm-up, never an audit transformation."""
import json
from pathlib import Path

WARMUP_EPOCHS = 1
WARMUP_LR = .0005
WARMUP_STEPS = 64


def warmup_rows(manifest, train):
    frozen = json.loads((Path(__file__).resolve().parent / "data/manifest.json").read_text())
    if manifest != frozen:
        raise ValueError("curriculum requires the unchanged frozen source manifest")
    original = [row for group in manifest["splits"]["train"] for row in group["pairs"]]
    if train != original:
        raise ValueError("curriculum accepts only the exact training split")
    documents = {doc["doi"]: doc for group in manifest["splits"]["train"] for doc in group["documents"]}
    result = []
    for row in train:
        own, donor = documents[row["source_id"]], documents[row["donor_id"]]
        if row["chosen"] != own["title"] or row["rejected"] != donor["title"]:
            raise ValueError("curriculum must retain the original human-published titles")
        prompt = "Give a concise title for this requested study's complete abstract:\n\n" + own["abstract"]
        result.append(dict(row, prompt=prompt, training_view="complete_requested_reference_warmup"))
    return result


def check_tokens(rows, tokenizer, render):
    checked, maximum = 0, 0
    for row in rows:
        for side in ("chosen", "rejected"):
            ids = tokenizer.encode(render(row["prompt"], row[side])).ids
            if not ids or len(ids) > 256:
                raise ValueError("complete curriculum source exceeds canonical token budget")
            maximum = max(maximum, len(ids))
            checked += 1
    return {"complete_warmup_inputs": checked, "maximum_warmup_tokens": maximum}


def source_check():
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "data/manifest.json").read_text())
    train = [row for g in manifest["splits"]["train"] for row in g["pairs"]]
    rows = warmup_rows(manifest, train)
    if len(rows) != 256 or len({r["group_id"] for r in rows}) != 128:
        raise ValueError("complete training curriculum required")
    return {"curriculum_source": "PASS", "training_pairs": len(rows), "validation_used": False, "audit_used": False}


if __name__ == "__main__":
    print(json.dumps(source_check()))
