# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Train-fit once remotely. Audit and reproduction consume frozen coefficients."""
import math
import re
from collections import Counter

from contracts import remote_only

WORDS = re.compile(r"[a-z0-9']+")
ALARM = {"negative", "flagged", "danger", "hate", "threat", "kill", "harm"}


def features(row, coefficients):
    idf = coefficients["idf"]
    def vector(text):
        return {w: n * idf[w] for w, n in Counter(WORDS.findall(text.casefold())).items() if w in idf}
    def cosine(a, b):
        den = math.sqrt(sum(x*x for x in a.values()) * sum(x*x for x in b.values()))
        return sum(x * b.get(w, 0) for w, x in a.items()) / den if den else 0.0
    a, b, p = vector(row["chosen"]), vector(row["rejected"]), vector(row["prompt"])
    def keyword(text):
        return sum(w in ALARM for w in WORDS.findall(text.casefold()))
    bow = coefficients["answer_only"]
    def answer_score(text):
        counts = Counter(WORDS.findall(text.casefold()))
        norm = math.sqrt(sum(counts[w] ** 2 for w in bow["weights"] if w in counts)) or 1.0
        return sum(n * bow["weights"].get(w, 0) for w, n in counts.items()) / norm
    return {"length": len(row["chosen"]) - len(row["rejected"]),
            "tfidf_overlap": cosine(p, a) - cosine(p, b),
            "alarm_keywords": keyword(row["chosen"]) - keyword(row["rejected"]),
            "answer_only": answer_score(row["chosen"]) - answer_score(row["rejected"])}


def scores(rows, coefficients):
    values = [features(r, coefficients) for r in rows]
    if not values:
        raise ValueError("empty shortcut evaluation")
    return {name: sum(0.5 if abs(r[name]) < 1e-10 else float(r[name] * direction > 0) for r in values) / len(values)
            for name, direction in coefficients["directions"].items()}


def fit(train):
    remote_only()
    import numpy as np
    documents = [set(WORDS.findall(r["prompt"].casefold())) for r in train]
    frequencies = Counter(w for d in documents for w in d)
    idf = {w: math.log((1 + len(train)) / (1 + n)) + 1 for w, n in sorted(frequencies.items())}
    counts = Counter()
    texts = [r["chosen"] for r in train] + [r["rejected"] for r in train]
    for text in texts:
        counts.update(set(WORDS.findall(text.casefold())))
    vocab = [w for w, _ in sorted(counts.items(), key=lambda r: (-r[1], r[0]))[:4000]]
    lookup = {w: i for i, w in enumerate(vocab)}
    x = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in WORDS.findall(text.casefold()):
            if word in lookup:
                x[i, lookup[word]] += 1
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    y = np.concatenate((np.ones(len(train)), np.zeros(len(train)))).astype(np.float32)
    w, b = np.zeros(len(vocab), dtype=np.float32), np.float32(0)
    rng = np.random.default_rng(42)
    for _ in range(60):
        order = rng.permutation(len(y))
        for start in range(0, len(y), 64):
            indices = order[start:start + 64]
            probability = 1 / (1 + np.exp(-np.clip(x[indices] @ w + b, -40, 40)))
            gradient = (probability - y[indices]) / len(indices)
            w -= 0.5 * (x[indices].T @ gradient + 1e-4 * w)
            b -= 0.5 * gradient.sum()
    coefficient = {"idf": idf, "answer_only": {"weights": {word: float(w[i]) for i, word in enumerate(vocab)}, "bias": float(b)},
                   "directions": {name: 1 for name in ("length", "tfidf_overlap", "alarm_keywords", "answer_only")},
                   "scope": "Training-only fitted coefficients. Frozen bytes are reused; audit never fits coefficients."}
    forward = scores(train, coefficient)
    coefficient["directions"] = {name: 1 if accuracy >= 0.5 else -1 for name, accuracy in forward.items()}
    return coefficient
