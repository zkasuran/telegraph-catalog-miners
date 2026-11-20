# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
import importlib.util
import json
import math
import sys
from collections import Counter

from data import HERE, digest, render
from frozen_baseline import fixed_features, words


def load_trainer(kit):
    source = kit / 'float_runtime/train_adapter.py'
    pins = json.loads((HERE / 'sources.json').read_text())
    if digest(source.read_bytes()) != pins['trainer_sha256']:
        raise ValueError('reference trainer pin mismatch')
    sys.path.insert(0, str(kit))
    spec = importlib.util.spec_from_file_location('qa_pinned_trainer', source)
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    return trainer

def pair_accuracy(values):
    return sum(1 if x > 0 else 0.5 if x == 0 else 0 for x in values) / len(values)

def fit_controls(train, tokenizer):
    """Fresh train-only diagnostic. Never selects or replaces the baseline."""
    import numpy as np
    df = Counter()
    for row in train:
        for field in ('prompt', 'chosen', 'rejected'):
            df.update(set(words(row[field])))
    vocab = {term: i for i, (term, _) in enumerate(sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))[:2000])}
    idf = np.array([math.log((1 + 3 * len(train)) / (1 + df[term])) + 1 for term in vocab])

    def vector(text):
        vec = np.zeros(len(vocab))
        for term in words(text):
            if term in vocab:
                vec[vocab[term]] += 1
        vec *= idf
        return vec / max(np.linalg.norm(vec), 1e-08)
    features = np.array([fixed_features(r['prompt'], r[s]) for r in train for s in ('chosen', 'rejected')], dtype=float)
    scale = np.maximum(features.std(axis=0), 1)
    x = np.array([vector(r[s]) for r in train for s in ('chosen', 'rejected')])
    y = np.tile([1.0, 0.0], len(train))

    def logistic(matrix):
        weight = np.zeros(matrix.shape[1])
        for _ in range(60):
            prediction = 1 / (1 + np.exp(-np.clip(matrix @ weight, -30, 30)))
            weight -= 0.2 * (matrix.T @ (prediction - y) / len(y) + 0.0001 * weight)
        return weight
    bow, style = (logistic(x), logistic(features / scale))
    orientations = {}
    for name, diffs in {'length': [len(r['chosen']) - len(r['rejected']) for r in train], 'token_length': [len(tokenizer.encode(render(r['prompt'], r['chosen'])).ids) - len(tokenizer.encode(render(r['prompt'], r['rejected'])).ids) for r in train], 'overlap': [float(vector(r['prompt']) @ (vector(r['chosen']) - vector(r['rejected']))) for r in train]}.items():
        orientations[name] = 1 if pair_accuracy(diffs) >= 0.5 else -1

    def scores(rows):
        result = {name: [] for name in ('length', 'token_length', 'overlap', 'answer_only', 'surface')}
        for row in rows:
            chosen, rejected = (vector(row['chosen']), vector(row['rejected']))
            result['length'].append(orientations['length'] * (len(row['chosen']) - len(row['rejected'])))
            result['token_length'].append(orientations['token_length'] * (len(tokenizer.encode(render(row['prompt'], row['chosen'])).ids) - len(tokenizer.encode(render(row['prompt'], row['rejected'])).ids)))
            result['overlap'].append(orientations['overlap'] * float(vector(row['prompt']) @ (chosen - rejected)))
            result['answer_only'].append(float((chosen - rejected) @ bow))
            difference = np.array(fixed_features(row['prompt'], row['chosen'])) - fixed_features(row['prompt'], row['rejected'])
            result['surface'].append(float(difference / scale @ style))
        return {name: pair_accuracy(values) for name, values in result.items()}
    frozen = {'vocabulary': vocab, 'idf': idf.tolist(), 'bow': bow.tolist(), 'style': style.tolist(), 'scale': scale.tolist(), 'orientations': orientations, 'fit_split': 'train'}
    return (scores, frozen)
