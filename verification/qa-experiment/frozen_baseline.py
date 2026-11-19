# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Immutable train-only shortcut replay. No fitting and no NumPy/BLAS scoring."""
import json
import math
import re
from collections import Counter
from types import MappingProxyType

from data import HERE, digest, render, write_json

MAX_BASELINE_BYTES = 1 << 20
FIELDS = {"vocabulary", "idf", "bow", "style", "scale", "orientations", "fit_split"}
WORDS = re.compile(r"[a-z0-9']+")
SOURCE = "OpenAssistant/oasst1"
REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
TOKENIZER_SHA = "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456"


def words(text):
    return WORDS.findall(text.casefold())


def fixed_features(prompt, answer):
    terms = words(answer)
    lower = answer.casefold()
    return [len(answer), len(terms), answer.count("\n"), answer.count("```"),
            int("http" in lower), sum(lower.count(x) for x in ("sorry", "cannot", "can't", "unable")),
            sum(c.isdigit() for c in answer), sum(c in "!?;:" for c in answer)]


def unique_object(items):
    result = {}
    for name, value in items:
        if name in result:
            raise ValueError("duplicate key in baseline artifact")
        result[name] = value
    return result


def decode_frozen(raw):
    if len(raw) > MAX_BASELINE_BYTES:
        raise ValueError("frozen baseline byte ceiling")
    return json.loads(raw, object_pairs_hook=unique_object)


def train_vocabulary(train):
    if len(train) != 128:
        raise ValueError("baseline identity requires exactly 128 training originals")
    df = Counter()
    for row in train:
        for field in ("prompt", "chosen", "rejected"):
            df.update(set(words(row[field])))
    vocab = {word: i for i, (word, _) in enumerate(sorted(df.items(), key=lambda item: (-item[1], item[0]))[:2000])}
    return df, vocab


def validate_frozen(frozen, train):
    if not isinstance(frozen, dict) or set(frozen) != FIELDS or frozen["fit_split"] != "train":
        raise ValueError("invalid frozen train-only baseline schema")
    df, expected = train_vocabulary(train)
    vocab = frozen["vocabulary"]
    if (not isinstance(vocab, dict) or not 1 <= len(vocab) <= 2000
            or any(not isinstance(word, str) or WORDS.fullmatch(word) is None or type(index) is not int for word, index in vocab.items())
            or sorted(vocab.values()) != list(range(len(vocab))) or vocab != expected):
        raise ValueError("frozen vocabulary does not match the exact training source")
    for field, size in (("idf", len(vocab)), ("bow", len(vocab)), ("style", 8), ("scale", 8)):
        values = frozen[field]
        if (not isinstance(values, list) or len(values) != size
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)):
            raise ValueError("frozen baseline dimension or finite-value check failed: " + field)
    if any(value < 1 for value in frozen["scale"]):
        raise ValueError("frozen feature scale must be positive and at least one")
    for word, index in vocab.items():
        expected_idf = math.log((1 + 3 * len(train)) / (1 + df[word])) + 1
        if not math.isclose(frozen["idf"][index], expected_idf, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("frozen IDF does not describe the training vocabulary")
    orientations = frozen["orientations"]
    if (not isinstance(orientations, dict) or set(orientations) != {"length", "token_length", "overlap"}
            or any(type(value) is not int or value not in (-1, 1) for value in orientations.values())):
        raise ValueError("invalid frozen train-only shortcut orientations")


def replay_controls(frozen, tokenizer):
    # Copies prevent callers from mutating the scorer's fixed baseline state.
    vocab = MappingProxyType(dict(frozen["vocabulary"]))
    idf, bow = tuple(frozen["idf"]), tuple(frozen["bow"])
    scale, style = tuple(frozen["scale"]), tuple(frozen["style"])
    orientations = MappingProxyType(dict(frozen["orientations"]))

    def vector(text):
        counts = Counter(words(text))
        values = [0.0] * len(vocab)
        for word, count in counts.items():
            if word in vocab:
                index = vocab[word]
                values[index] = count * idf[index]
        denominator = max(math.sqrt(math.fsum(value * value for value in values)), 1e-8)
        return tuple(value / denominator for value in values)

    def scores(rows):
        if not rows:
            raise ValueError("empty frozen-control evaluation")
        totals = {name: 0.0 for name in ("length", "token_length", "overlap", "answer_only", "surface")}
        for row in rows:
            chosen, rejected, prompt = vector(row["chosen"]), vector(row["rejected"]), vector(row["prompt"])
            difference = tuple(left - right for left, right in zip(chosen, rejected))
            chosen_style, rejected_style = fixed_features(row["prompt"], row["chosen"]), fixed_features(row["prompt"], row["rejected"])
            values = {
                "length": orientations["length"] * (len(row["chosen"]) - len(row["rejected"])),
                "token_length": orientations["token_length"] * (len(tokenizer.encode(render(row["prompt"], row["chosen"])).ids)
                                     - len(tokenizer.encode(render(row["prompt"], row["rejected"])).ids)),
                "overlap": orientations["overlap"] * math.fsum(left * right for left, right in zip(prompt, difference)),
                "answer_only": math.fsum(left * right for left, right in zip(difference, bow)),
                "surface": math.fsum((left - right) / scaling * weight for left, right, scaling, weight in
                                     zip(chosen_style, rejected_style, scale, style)),
            }
            for name, value in values.items():
                if not math.isfinite(value):
                    raise ValueError("nonfinite frozen baseline score")
                totals[name] += 1 if value > 0 else .5 if value == 0 else 0
        return {name: total / len(rows) for name, total in totals.items()}

    return scores


def load_frozen_controls(root, train, tokenizer):
    contract = json.loads((HERE / "contract.json").read_text())
    bundle = HERE / "baselines"
    manifest = json.loads((bundle / "manifest.json").read_text())
    expected_hash = contract["freeze_sha256"]["frozen-controls.json"]
    expected_identity = {"schema_version": 1, "artifact": "frozen-controls.json", "artifact_sha256": expected_hash,
                         "fit_split": "train", "training_rows": 128, "training_sha256": contract["splits"]["train"]["sha256"],
                         "source": SOURCE, "source_revision": REVISION,
                         "source_manifest_sha256": contract["freeze_sha256"]["sources.json"], "tokenizer_sha256": TOKENIZER_SHA}
    if any(manifest.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("frozen baseline manifest training/source identity mismatch")
    for path, sha in ((root / "train.jsonl", expected_identity["training_sha256"]),
                      (HERE / "sources.json", expected_identity["source_manifest_sha256"]),
                      (root / "sources/tokenizer.json", TOKENIZER_SHA)):
        if digest(path.read_bytes()) != sha:
            raise ValueError("frozen baseline input identity mismatch: " + path.name)
    stored_train = [json.loads(line) for line in (root / "train.jsonl").read_text().split("\n") if line]
    if train != stored_train or any(row.get("source_domain") != SOURCE or row.get("source_revision") != REVISION for row in train):
        raise ValueError("baseline replay was not given the exact training originals")
    path = bundle / "frozen-controls.json"
    if path.is_symlink() or path.stat().st_size > MAX_BASELINE_BYTES:
        raise ValueError("frozen baseline must be a bounded regular artifact")
    raw = path.read_bytes()
    if digest(raw) != expected_hash:
        raise ValueError("frozen baseline artifact hash mismatch")
    frozen = decode_frozen(raw)
    validate_frozen(frozen, train)
    destination = root / "frozen-controls.json"
    if destination.is_file() and digest(destination.read_bytes()) != expected_hash:
        raise ValueError("existing frozen artifact was changed, not an accepted baseline")
    if not destination.exists():
        destination.write_bytes(raw)
    write_json(root / "baseline-replay.json", {"status": "VALIDATED_IMMUTABLE_REPLAY", **expected_identity,
               "scoring": "fixed scalar fsum, no NumPy/BLAS reductions", "refit": False,
               "audit_scored": False, "original_failed_refit": {"run_id": "44463468482", "commit": "f29434c4af4d68d79aae912202118847d862e571", "status": "FAILED, not reclassified"}})
    return replay_controls(frozen, tokenizer), frozen
