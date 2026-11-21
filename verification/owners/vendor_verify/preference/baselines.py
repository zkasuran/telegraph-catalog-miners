# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Train-fitted immutable classical shortcut probes, no neural imports.

Any fitted vocabulary retains HelpSteer2 CC BY 4.0. Hexadecimal binary64
coefficients reproduce the frozen artifact without floating-point refitting.
"""

import json
import math
import re
from collections import Counter

from data import FIELDS, encode, identity, sha

WORDS = re.compile(r"[a-z0-9']+")
FEATURES = ("length", "digits", "tfidf_overlap", "answer_only", "train_answer_lookup")
MAX_TRAIN_BYTES = 4 << 20


def differences(row, coefficients):
    idf = {w: float.fromhex(v) for w, v in coefficients["idf_hex"].items()}
    odds = coefficients["answer_log_odds_hex"]

    def vector(text):
        return {
            w: n * idf[w]
            for w, n in Counter(WORDS.findall(text.casefold())).items()
            if w in idf
        }

    def cosine(a, b):
        denominator = math.sqrt(
            math.fsum(v * v for v in a.values()) * math.fsum(v * v for v in b.values())
        )
        return (
            math.fsum(a[w] * b.get(w, 0) for w in sorted(a)) / denominator
            if denominator
            else 0.0
        )

    def answer_score(text):
        words = WORDS.findall(text.casefold())
        return math.fsum(float.fromhex(odds.get(w, "0x0.0p+0")) for w in words) / max(
            1, len(words)
        )

    def lookup(text):
        return coefficients["answer_labels"].get(identity(text), 0)

    prompt, good, bad = (vector(row[k]) for k in FIELDS)
    return {
        "length": len(row["chosen"]) - len(row["rejected"]),
        "digits": sum(c.isdigit() for c in row["chosen"])
        - sum(c.isdigit() for c in row["rejected"]),
        "tfidf_overlap": cosine(prompt, good) - cosine(prompt, bad),
        "answer_only": answer_score(row["chosen"]) - answer_score(row["rejected"]),
        "train_answer_lookup": lookup(row["chosen"]) - lookup(row["rejected"]),
    }


def measure(rows, coefficients):
    if not rows:
        raise ValueError("empty baseline measurement")
    values = [differences(row, coefficients) for row in rows]
    result = {}
    for name in FEATURES:
        direction = coefficients["directions"][name]
        scores = [v[name] * direction for v in values]
        result[name] = {
            "accuracy": sum(0.5 if abs(v) < 1e-10 else float(v > 0) for v in scores)
            / len(scores),
            "ties": sum(abs(v) < 1e-10 for v in scores),
            "pairs": len(scores),
        }
    return result


def fit_training(raw):
    if not raw or len(raw) > MAX_TRAIN_BYTES:
        raise ValueError("baseline training byte ceiling")
    rows = [json.loads(line) for line in raw.decode().split("\n") if line.strip()]
    for row in rows:
        if not isinstance(row, dict) or not all(
            isinstance(row.get(k), str) and row[k] for k in FIELDS
        ):
            raise ValueError("invalid baseline training row")
    frequencies = Counter(
        w for row in rows for w in set(WORDS.findall(row["prompt"].casefold()))
    )
    good, bad, labels = Counter(), Counter(), {}
    for row in rows:
        good.update(WORDS.findall(row["chosen"].casefold()))
        bad.update(WORDS.findall(row["rejected"].casefold()))
        for side, label in (("chosen", 1), ("rejected", -1)):
            key = identity(row[side])
            if key in labels and labels[key] != label:
                labels[key] = 0
            else:
                labels[key] = label
    vocabulary = sorted(set(good) | set(bad))
    ng, nb = sum(good.values()) + len(vocabulary), sum(bad.values()) + len(vocabulary)
    coefficients = {
        "schema_version": 1,
        "family": "VENDOR_VERIFY",
        "source_licence": "CC-BY-4.0",
        "attribution": "NVIDIA HelpSteer2, Wang et al. 2024",
        "fit_scope": "train only, general response-quality shortcut probes",
        "training_sha256": sha(raw),
        "training_pairs": len(rows),
        "idf_hex": {
            w: float.hex(math.log((1 + len(rows)) / (1 + frequencies[w])) + 1)
            for w in sorted(frequencies)
        },
        "answer_log_odds_hex": {
            w: float.hex(math.log((good[w] + 1) / ng) - math.log((bad[w] + 1) / nb))
            for w in vocabulary
        },
        "answer_labels": labels,
        "directions": dict.fromkeys(FEATURES, 1),
    }
    metrics = measure(rows, coefficients)
    coefficients["directions"] = {
        k: 1 if metrics[k]["accuracy"] >= 0.5 else -1 for k in FEATURES
    }
    return coefficients


def load_frozen(raw, training_sha256):
    if not raw or len(raw) > MAX_TRAIN_BYTES:
        raise ValueError("baseline artifact byte ceiling")
    value = json.loads(raw)
    if (
        value.get("family") != "VENDOR_VERIFY"
        or value.get("schema_version") != 1
        or value.get("training_sha256") != training_sha256
    ):
        raise ValueError("baseline training identity mismatch")
    if set(value.get("directions", {})) != set(FEATURES) or any(
        type(d) is not int or d not in {-1, 1} for d in value["directions"].values()
    ):
        raise ValueError("invalid baseline directions")
    for key in ("idf_hex", "answer_log_odds_hex"):
        if not isinstance(value.get(key), dict) or any(
            not isinstance(v, str) or not math.isfinite(float.fromhex(v))
            for v in value[key].values()
        ):
            raise ValueError("nonfinite baseline coefficient")
    if not isinstance(value.get("answer_labels"), dict) or any(
        type(v) is not int or v not in {-1, 0, 1}
        for v in value["answer_labels"].values()
    ):
        raise ValueError("invalid training lookup labels")
    return value


def freeze_once(training_path, output_path):
    if training_path.name != "train.jsonl":
        raise ValueError("only train.jsonl may fit coefficients")
    if output_path.exists():
        raise ValueError("frozen coefficients already retained")
    raw = training_path.read_bytes()
    encoded = encode(fit_training(raw))
    with output_path.open("xb") as target:
        target.write(encoded)
    return sha(encoded)
