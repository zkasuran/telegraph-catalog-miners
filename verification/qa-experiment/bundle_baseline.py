# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Bundle original verified train-only coefficient bytes without refitting."""
import argparse
import json
from pathlib import Path

from data import HERE, digest, write_json
from frozen_baseline import (
    REVISION,
    SOURCE,
    TOKENIZER_SHA,
    decode_frozen,
    validate_frozen,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads((HERE / "contract.json").read_text())
    if digest((HERE / "sources.json").read_bytes()) != contract["freeze_sha256"]["sources.json"]:
        raise ValueError("original baseline source manifest changed")
    raw = args.original.read_bytes()
    if digest(raw) != contract["freeze_sha256"]["frozen-controls.json"]:
        raise ValueError("original frozen coefficient artifact does not match the contract")
    train = (args.data / "train.jsonl").read_bytes()
    if digest(train) != contract["splits"]["train"]["sha256"]:
        raise ValueError("original baseline training identity mismatch")
    rows = [json.loads(line) for line in train.decode().split("\n") if line]
    validate_frozen(decode_frozen(raw), rows)
    output = HERE / "baselines"
    output.mkdir(exist_ok=True)
    (output / "frozen-controls.json").write_bytes(raw)
    write_json(output / "manifest.json", {"schema_version": 1, "artifact": "frozen-controls.json", "artifact_sha256": digest(raw),
               "fit_split": "train", "training_rows": 128, "training_sha256": digest(train), "source": SOURCE,
               "source_revision": REVISION, "source_manifest_sha256": contract["freeze_sha256"]["sources.json"],
               "tokenizer_sha256": TOKENIZER_SHA, "origin": "original verified source-only QA control fit",
               "licence": "LicenseRef-zkasuran-SAND-1.0 on own coefficients; original source tokens/annotations retain Apache-2.0",
               "terms": "../LICENSE, ../UPSTREAM-APACHE-2.0.txt, ../KIT-LICENSE and ../NOTICE"})
    print("Bundled immutable baseline:", digest(raw), "training rows:", len(rows))


if __name__ == "__main__":
    main()
