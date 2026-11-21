# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Prepare bounded general preferences without reading evaluation-pool material.

Source text retains NVIDIA HelpSteer2 CC BY 4.0. Source annotations are not
verified VAT facts. This experiment cannot issue a registration-ready report.
"""

import argparse
import gzip
import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_COMMIT = "990b2711a36180dd19d9c94b8627844866f8982a"
SOURCE_SHA256 = "a5cd48600fb7a330cf0ccc8f59051e24e8f236907c379f42eff1ba18da55204b"
SOURCE_BYTES = 15339272
CARD_SHA256 = "835effb9e7d9cd0e8b7b8c1816d1d97a8961a036543108e4a0b6a5e22712ff7b"
LEGAL_SHA256 = "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411"
COUNTS = {"train": 128, "validation": 64, "audit": 64}
SALT = "vendor-general-preference-v1:"
MAX_COMPRESSED = 32 << 20
MAX_EXPANDED = 256 << 20
MAX_LINE = 1 << 20
MAX_PAIRS = 10000
MAX_RENDER_BYTES = 1536
MAX_SECONDS = 300
SENSITIVE = re.compile(
    r"private[ _-]?key|api[ _-]?key|access[ _-]?token|password|credential|seed phrase",
    re.IGNORECASE,
)
FIELDS = ("prompt", "chosen", "rejected")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def identity(text):
    return sha(normalized(text).encode())


def encode(value):
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()


def render(prompt, answer):
    # Literal primary train_adapter.render / node RenderForJudgment contract.
    return f"### Task:\n{prompt}\n\n### Answer:\n{answer}"


def bounded_read(path, limit):
    path = Path(path)
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("symlink input refused")
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError("input size/type ceiling")
    with path.open("rb") as source:
        raw = source.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("input grew beyond ceiling")
    return raw


def direct_preference(record):
    if not isinstance(record, dict) or record.get("split") not in {
        "train",
        "val",
        "validation",
    }:
        raise ValueError("original partition required")
    # Never access validation text or annotations, even when malformed.
    if record["split"] != "train":
        return None, "official_validation_excluded"
    if not all(
        isinstance(record.get(k), str) and record[k].strip()
        for k in ("prompt", "response_1", "response_2")
    ):
        return None, "invalid_text_excluded"
    strength = record.get("preference_strength")
    if type(strength) not in {int, float} or not math.isfinite(strength):
        return None, "invalid_strength_excluded"
    if strength in {0, -100}:
        return None, "neutral_or_neither_valid_excluded"
    if not -3 <= strength <= 3:
        return None, "invalid_strength_excluded"
    annotations = record.get("three_most_similar_preferences")
    if not isinstance(annotations, list) or len(annotations) != 3:
        return None, "missing_three_annotations_excluded"
    values = [a.get("strength") if isinstance(a, dict) else None for a in annotations]
    if not all(
        type(v) in {int, float} and math.isfinite(v) and -3 <= v <= 3 for v in values
    ):
        return None, "invalid_annotation_excluded"
    if max(values) - min(values) > 2:
        return None, "annotation_spread_excluded"
    if strength * sum(values) <= 0:
        return None, "annotation_direction_disagreement_excluded"
    if identity(record["response_1"]) == identity(record["response_2"]):
        return None, "normalized_identical_answers_excluded"
    selected = 1 if strength < 0 else 2
    pair = {
        "prompt": record["prompt"],
        "chosen": record[f"response_{selected}"],
        "rejected": record[f"response_{3 - selected}"],
        "source_strength": strength,
        "source_preferred_response": selected,
        "label_kind": "preference_pair",
        "margin": 0.0,
    }
    if any(SENSITIVE.search(pair[k]) for k in FIELDS):
        return None, "credential_topic_excluded"
    return pair, "usable_direct_preference"


def read_source(path, expected_sha=SOURCE_SHA256, expected_bytes=SOURCE_BYTES):
    raw = bounded_read(path, MAX_COMPRESSED)
    if sha(raw) != expected_sha or len(raw) != expected_bytes:
        raise ValueError("immutable original source hash/size mismatch")
    counts, pairs = Counter(), []
    expanded = 0
    start = time.monotonic()
    with gzip.GzipFile(fileobj=__import__("io").BytesIO(raw)) as source:
        while line := source.readline(MAX_LINE + 1):
            expanded += len(line)
            if (
                len(line) > MAX_LINE
                or expanded > MAX_EXPANDED
                or time.monotonic() - start > MAX_SECONDS
            ):
                raise ValueError("source expansion/time ceiling")
            counts["records"] += 1
            row = json.loads(line)
            pair, reason = direct_preference(row)
            counts[reason] += 1
            if pair is not None:
                pairs.append(
                    {
                        **pair,
                        "source_row": counts["records"],
                        "source_row_sha256": sha(line),
                    }
                )
            if len(pairs) > MAX_PAIRS:
                raise ValueError("source pair ceiling")
    return pairs, dict(counts)


def grouped(pairs):
    """Union all reused prompts/answers before any length filter or allocation."""
    if len(pairs) > MAX_PAIRS:
        raise ValueError("grouping pair ceiling")
    parent = list(range(len(pairs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, pair in enumerate(pairs):
        for field in FIELDS:
            key = identity(pair[field])
            if key in owners:
                a, b = find(i), find(owners[key])
                parent[max(a, b)] = min(a, b)
            owners[key] = i
    components = defaultdict(list)
    for i, pair in enumerate(pairs):
        components[find(i)].append(pair)
    pool = []
    removed = Counter()
    for members in components.values():
        orientations = defaultdict(set)
        for pair in members:
            good, bad = identity(pair["chosen"]), identity(pair["rejected"])
            orientations[tuple(sorted((good, bad)))].add((good, bad))
        if any(len(v) > 1 for v in orientations.values()):
            removed["conflicting_components"] += 1
            continue
        group = sha(encode(sorted({identity(p[k]) for p in members for k in FIELDS})))
        # Select without using annotation magnitude, length direction or model scores.
        eligible = [
            p
            for p in members
            if all(
                len(render(p["prompt"], p[k]).encode()) <= MAX_RENDER_BYTES
                for k in ("chosen", "rejected")
            )
        ]
        if not eligible:
            removed["oversize_components"] += 1
            continue
        row = min(eligible, key=lambda p: p["source_row_sha256"])
        pool.append({**row, "group_id": group, "component_originals": len(members)})
    return sorted(pool, key=lambda p: sha((SALT + p["group_id"]).encode())), dict(
        removed
    )


def check_splits(splits):
    seen_groups, seen_text = set(), set()
    for split, rows in splits.items():
        if not rows:
            raise ValueError("empty split")
        groups, texts = set(), set()
        for row in rows:
            if not all(
                isinstance(row.get(k), str) and row[k].strip()
                for k in (*FIELDS, "group_id")
            ):
                raise ValueError("invalid split row")
            if identity(row["chosen"]) == identity(row["rejected"]):
                raise ValueError("identical normalized pair")
            if row["group_id"] in groups:
                raise ValueError("repeated source group")
            groups.add(row["group_id"])
            texts.update(identity(row[k]) for k in FIELDS)
        if groups & seen_groups or texts & seen_text:
            raise ValueError("normalized source/text overlap")
        seen_groups.update(groups)
        seen_text.update(texts)
    return True


def allocate(pool, counts=None):
    counts = COUNTS if counts is None else counts
    if set(counts) != set(COUNTS) or any(
        type(n) is not int or n < 1 for n in counts.values()
    ):
        raise ValueError("invalid split allocation")
    if len(pool) < sum(counts.values()):
        raise ValueError("insufficient independent groups, no padding")
    offset, splits = 0, {}
    for split in ("train", "validation", "audit"):
        splits[split] = pool[offset : offset + counts[split]]
        offset += counts[split]
    check_splits(splits)
    return splits


def prepare(source, card, legal, out, frozen_baselines=None):
    # Refuse repeat work before parsing the source or fitting any coefficients.
    out = Path(out)
    if out.exists():
        raise ValueError("retained experiment output already exists")
    card_raw = bounded_read(card, 65536)
    legal_raw = bounded_read(legal, 65536)
    if sha(card_raw) != CARD_SHA256 or sha(legal_raw) != LEGAL_SHA256:
        raise ValueError("retained primary card/grant identity mismatch")
    pairs, counts = read_source(source)
    pool, removed = grouped(pairs)
    splits = allocate(pool)
    from baselines import fit_training, load_frozen, measure

    encoded = {s: b"".join(encode(row) for row in rows) for s, rows in splits.items()}
    coefficients = (
        load_frozen(bounded_read(frozen_baselines, 4 << 20), sha(encoded["train"]))
        if frozen_baselines is not None
        else fit_training(encoded["train"])
    )
    # No audit evaluation, including shortcut baselines, during development.
    metrics = {s: measure(splits[s], coefficients) for s in ("train", "validation")}
    report = {
        "schema_version": 1,
        "family": "VENDOR_VERIFY",
        "status": "GENERAL_PREFERENCE_EXPERIMENT_PREPARED",
        "scope": "General response-quality development. Not an exact Vendor or VIES audit",
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "source_card_sha256": CARD_SHA256,
        "source_grant_sha256": LEGAL_SHA256,
        "source_licence": "CC-BY-4.0",
        "attribution": "NVIDIA HelpSteer2, Wang et al. 2024, arXiv:2410.01257",
        "source_counts": counts,
        "eligible_independent_components": len(pool),
        "exclusions": removed,
        "allocation": "Frozen v1 salted hash order, one original per connected prompt/answer component",
        "normalized_overlap": 0,
        "audit_evaluated": False,
        "exact_vendor_audit_admitted": False,
        "registration_ready": False,
        "model_operations": False,
        "files": {
            s: {"sha256": sha(raw), "pairs": len(splits[s]), "bytes": len(raw)}
            for s, raw in encoded.items()
        },
        "baseline_sha256": sha(encode(coefficients)),
        "development_baselines": metrics,
        "limits": [
            "Prompt/answer connected components are not independently confirmed entity/time/source-domain groups",
            "Source annotations are human preferences over mainly model-generated responses, not fact-checked VAT labels",
            "No synthetic preferences or duplicated rating representations used",
            "Byte-length screen is not tokenizer retention; remote exact 512-token check required",
            "Audit originals sealed for this general experiment, not a qualified task audit",
        ],
    }
    out.mkdir(mode=0o700, parents=True)
    for split, raw in encoded.items():
        with (out / (split + ".jsonl")).open("xb") as target:
            target.write(raw)
    for name, raw in (
        ("BASELINES.json", encode(coefficients)),
        ("SOURCE-CARD.md", card_raw),
        ("CC-BY-4.0.txt", legal_raw),
        ("DATA-REPORT.json", encode(report)),
    ):
        with (out / name).open("xb") as target:
            target.write(raw)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--card", required=True, type=Path)
    parser.add_argument("--legal", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = prepare(args.source, args.card, args.legal, args.out)
    # Only metadata and aggregate counts leave this CLI.
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
