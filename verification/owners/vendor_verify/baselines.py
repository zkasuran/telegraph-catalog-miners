# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Immutable training-only surface coefficients. No neural imports or calls.

Source vocabulary retains Brreg NLOD-2.0. Coefficients are serialized as exact
binary64 hexadecimal strings once, not refitted during report reproduction.
"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

WORDS = re.compile(r"[a-z0-9']+")
NAMES = re.compile(r"[^\W\d_]+")
ALARM = ("danger", "flagged", "harm", "hate", "kill", "negative", "threat")
FEATURES = (
    "length",
    "tfidf_overlap",
    "alarm_keywords",
    "company_name_tokens",
    "digits",
)
MAX_BYTES = 2 << 20


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def decode_rows(raw):
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("empty or oversized baseline training input")
    rows = [json.loads(line) for line in raw.decode().split("\n") if line.strip()]
    if not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError("baseline rows must be objects")
    for row in rows:
        if any(
            not isinstance(row.get(k), str) or not row[k]
            for k in ("prompt", "chosen", "rejected")
        ):
            raise ValueError("missing baseline pair text")
        chosen = json.loads(row["chosen"])
        if (
            row.get("label") not in ("true", "false")
            or type(chosen.get("valid")) is not bool
        ):
            raise ValueError("missing baseline source state")
        if str(chosen["valid"]).lower() != row["label"] or not isinstance(
            chosen.get("name"), str
        ):
            raise ValueError("inconsistent baseline label")
    return rows


def values(row, coefficients):
    idf = {k: float.fromhex(v) for k, v in coefficients["idf_hex"].items()}
    odds = coefficients["name_log_odds_hex"]

    def vector(text):
        counts = Counter(WORDS.findall(text.casefold()))
        return {w: n * idf[w] for w, n in counts.items() if w in idf}

    def cosine(a, b):
        denominator = math.sqrt(
            math.fsum(v * v for v in a.values()) * math.fsum(v * v for v in b.values())
        )
        numerator = math.fsum(a[w] * b.get(w, 0) for w in sorted(a))
        return numerator / denominator if denominator else 0.0

    chosen = json.loads(row["chosen"])
    prompt, good, bad = (
        vector(row["prompt"]),
        vector(row["chosen"]),
        vector(row["rejected"]),
    )
    name_value = math.fsum(
        float.fromhex(odds.get(w, "0x0.0p+0"))
        for w in sorted(set(NAMES.findall(chosen["name"].casefold())))
    )
    return {
        "length": len(row["chosen"]) - len(row["rejected"]),
        "tfidf_overlap": cosine(prompt, good) - cosine(prompt, bad),
        "alarm_keywords": sum(
            w in ALARM for w in WORDS.findall(row["chosen"].casefold())
        )
        - sum(w in ALARM for w in WORDS.findall(row["rejected"].casefold())),
        "company_name_tokens": name_value * (1 if chosen["valid"] else -1),
        "digits": sum(c.isdigit() for c in row["chosen"])
        - sum(c.isdigit() for c in row["rejected"]),
    }


def accuracy(rows, name, direction, coefficients):
    differences = [values(r, coefficients)[name] * direction for r in rows]
    return sum(0.5 if abs(v) < 1e-10 else float(v > 0) for v in differences) / len(rows)


def fit_training(raw):
    rows = decode_rows(raw)
    frequency = Counter(
        w for r in rows for w in set(WORDS.findall(r["prompt"].casefold()))
    )
    names = {label: Counter() for label in ("true", "false")}
    for row in rows:
        names[row["label"]].update(
            set(NAMES.findall(json.loads(row["chosen"])["name"].casefold()))
        )
    all_names = set(names["true"]) | set(names["false"])
    coefficients = {
        "schema_version": 1,
        "family": "VENDOR_VERIFY",
        "source_licence": "NLOD-2.0",
        "attribution": "Brønnøysundregistrene, https://data.brreg.no/enhetsregisteret/api/enheter",
        "own_additions_licence": "LicenseRef-zkasuran-SAND-1.0",
        "training_sha256": digest(raw),
        "training_originals": len(rows),
        "fit_scope": "training only; classical surface baselines, not a neural model or independent audit",
        "idf_hex": {
            w: float.hex(math.log((1 + len(rows)) / (1 + frequency[w])) + 1)
            for w in sorted(frequency)
        },
        "name_log_odds_hex": {
            w: float.hex(math.log((names["true"][w] + 1) / (names["false"][w] + 1)))
            for w in sorted(all_names)
        },
    }
    coefficients["directions"] = {
        n: 1 if accuracy(rows, n, 1, coefficients) >= 0.5 else -1 for n in FEATURES
    }
    return coefficients


def freeze_once(training_path, output_path):
    if training_path.name != "train.jsonl":
        raise ValueError("coefficient fitting accepts train.jsonl only")
    raw = training_path.read_bytes()
    value = fit_training(raw)
    encoded = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()
    # Refuse to overwrite previous fitted coefficients, even with the same data.
    with output_path.open("xb") as out:
        out.write(encoded)
    return digest(encoded)


def load_coefficients(path, training_sha256):
    if path.is_symlink() or path.stat().st_size > MAX_BYTES:
        raise ValueError("unsafe baseline artifact")
    raw = path.read_bytes()
    value = json.loads(raw)
    if (
        value.get("schema_version") != 1
        or value.get("family") != "VENDOR_VERIFY"
        or value.get("training_sha256") != training_sha256
    ):
        raise ValueError("stale or cross-family baseline coefficients")
    if set(value.get("directions", {})) != set(FEATURES) or any(
        type(v) is not int or v not in (-1, 1) for v in value["directions"].values()
    ):
        raise ValueError("invalid frozen baseline directions")
    for name in ("idf_hex", "name_log_odds_hex"):
        if not isinstance(value.get(name), dict) or any(
            not isinstance(v, str) or not math.isfinite(float.fromhex(v))
            for v in value[name].values()
        ):
            raise ValueError("nonfinite baseline coefficient")
    return value, digest(raw)


def evaluate_frozen(splits, coefficients):
    if "audit" in splits or "test" in splits:
        raise ValueError("closed audit must not enter baseline selection/evaluation")
    return {
        "train_fitted_direction": coefficients["directions"],
        "by_split": {
            s: {
                n: round(accuracy(rows, n, d, coefficients), 4)
                for n, d in coefficients["directions"].items()
            }
            for s, rows in splits.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "coefficients_sha256": freeze_once(args.train, args.out),
                "fit_scope": "train only",
            }
        )
    )


if __name__ == "__main__":
    main()
