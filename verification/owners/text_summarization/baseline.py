# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Replay initial train-count coefficients. Never fit to reproduce machine bytes."""
import json
import math
import re
from collections import Counter
from types import MappingProxyType

from data import HERE, digest, render

WORDS = re.compile(r"[a-z0-9']+")


def load_controls(train, tokenizer):
    contract = json.loads((HERE / "contract.json").read_text())
    raw = (HERE / "data/frozen-count-baselines.json").read_bytes()
    if digest(raw) != contract["baseline_sha256"]:
        raise ValueError("immutable baseline bytes changed")
    c = json.loads(raw)
    if (c["fit_split"] != "train" or c["training_pairs"] != len(train)
            or c["training_sha256"] != contract["splits"]["train"]["sha256"]
            or c["source_manifest_sha256"] != contract["source_manifest_sha256"]):
        raise ValueError("baseline source/training identity mismatch")
    vocab = MappingProxyType(dict(c["vocabulary"]))
    idf, weights = tuple(c["idf"]), tuple(c["answer_only_count_logodds"])
    directions = MappingProxyType(dict(c["directions"]))
    if (sorted(vocab.values()) != list(range(len(idf))) or len(weights) != len(idf)
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in idf + weights)
            or any(x not in (-1, 1) for x in directions.values())):
        raise ValueError("invalid baseline dimensions or finite values")

    def vector(text):
        count = Counter(WORDS.findall(text.casefold()))
        values = [0.0] * len(idf)
        for word, n in count.items():
            if word in vocab:
                i = vocab[word]
                values[i] = n * idf[i]
        denominator = max(math.sqrt(math.fsum(x * x for x in values)), 1e-8)
        return tuple(x / denominator for x in values)

    def score(rows):
        if not rows:
            raise ValueError("empty baseline evaluation")
        totals = {name: 0.0 for name in directions}
        for row in rows:
            a, b, p = vector(row["chosen"]), vector(row["rejected"]), vector(row["prompt"])
            difference = tuple(x - y for x, y in zip(a, b))
            values = {"length": len(row["chosen"]) - len(row["rejected"]),
                      "token_length": len(tokenizer.encode(render(row["prompt"], row["chosen"])).ids) - len(tokenizer.encode(render(row["prompt"], row["rejected"])).ids),
                      "tfidf_overlap": math.fsum(x * y for x, y in zip(p, difference)),
                      "answer_only_counts": math.fsum(x * y for x, y in zip(difference, weights))}
            for name, value in values.items():
                value *= directions[name]
                totals[name] += 1 if value > 0 else .5 if value == 0 else 0
        return {name: total / len(rows) for name, total in totals.items()}

    return score
