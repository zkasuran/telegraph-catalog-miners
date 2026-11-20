# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
import json
import math
import re
from collections import Counter
from types import MappingProxyType

from data import render

MAX_BASELINE_BYTES = 1 << 20
FIELDS = {'vocabulary','idf','bow','style','scale','orientations','fit_split'}
WORDS = re.compile(r"[a-z0-9']+")

def words(text):
    return WORDS.findall(text.casefold())

def fixed_features(prompt, answer):
    terms = words(answer)
    lower = answer.casefold()
    return [len(answer), len(terms), answer.count('\n'), answer.count('```'), int('http' in lower), sum(lower.count(x) for x in ('sorry', 'cannot', "can't", 'unable')), sum(c.isdigit() for c in answer), sum(c in '!?;:' for c in answer)]

def unique_object(items):
    result = {}
    for name, value in items:
        if name in result:
            raise ValueError('duplicate key in baseline artifact')
        result[name] = value
    return result

def decode_frozen(raw):
    if len(raw) > MAX_BASELINE_BYTES:
        raise ValueError('frozen baseline byte ceiling')
    return json.loads(raw, object_pairs_hook=unique_object)

def train_vocabulary(train):
    if not train:
        raise ValueError('baseline requires training originals')
    df = Counter()
    for row in train:
        for field in ('prompt', 'chosen', 'rejected'):
            df.update(set(words(row[field])))
    vocab = {word: i for i, (word, _) in enumerate(sorted(df.items(), key=lambda item: (-item[1], item[0]))[:2000])}
    return (df, vocab)

def validate_frozen(frozen, train):
    if not isinstance(frozen, dict) or set(frozen) != FIELDS or frozen['fit_split'] != 'train':
        raise ValueError('invalid frozen train-only baseline schema')
    df, expected = train_vocabulary(train)
    vocab = frozen['vocabulary']
    if not isinstance(vocab, dict) or not 1 <= len(vocab) <= 2000 or any((not isinstance(word, str) or WORDS.fullmatch(word) is None or type(index) is not int for word, index in vocab.items())) or (sorted(vocab.values()) != list(range(len(vocab)))) or (vocab != expected):
        raise ValueError('frozen vocabulary does not match the exact training source')
    for field, size in (('idf', len(vocab)), ('bow', len(vocab)), ('style', 8), ('scale', 8)):
        values = frozen[field]
        if not isinstance(values, list) or len(values) != size or any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError('frozen baseline dimension or finite-value check failed: ' + field)
    if any(value < 1 for value in frozen['scale']):
        raise ValueError('frozen feature scale must be positive and at least one')
    for word, index in vocab.items():
        expected_idf = math.log((1 + 3 * len(train)) / (1 + df[word])) + 1
        if not math.isclose(frozen['idf'][index], expected_idf, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('frozen IDF does not describe the training vocabulary')
    orientations = frozen['orientations']
    if not isinstance(orientations, dict) or set(orientations) != {'length', 'token_length', 'overlap'} or any(type(value) is not int or value not in (-1, 1) for value in orientations.values()):
        raise ValueError('invalid frozen train-only shortcut orientations')

def replay_controls(frozen, tokenizer):
    vocab = MappingProxyType(dict(frozen['vocabulary']))
    idf, bow = (tuple(frozen['idf']), tuple(frozen['bow']))
    scale, style = (tuple(frozen['scale']), tuple(frozen['style']))
    orientations = MappingProxyType(dict(frozen['orientations']))

    def vector(text):
        counts = Counter(words(text))
        values = [0.0] * len(vocab)
        for word, count in counts.items():
            if word in vocab:
                index = vocab[word]
                values[index] = count * idf[index]
        denominator = max(math.sqrt(math.fsum(value * value for value in values)), 1e-08)
        return tuple(value / denominator for value in values)

    def scores(rows):
        if not rows:
            raise ValueError('empty frozen-control evaluation')
        totals = {name: 0.0 for name in ('length', 'token_length', 'overlap', 'answer_only', 'surface')}
        for row in rows:
            chosen, rejected, prompt = (vector(row['chosen']), vector(row['rejected']), vector(row['prompt']))
            difference = tuple((left - right for left, right in zip(chosen, rejected)))
            chosen_style, rejected_style = (fixed_features(row['prompt'], row['chosen']), fixed_features(row['prompt'], row['rejected']))
            values = {'length': orientations['length'] * (len(row['chosen']) - len(row['rejected'])), 'token_length': orientations['token_length'] * (len(tokenizer.encode(render(row['prompt'], row['chosen'])).ids) - len(tokenizer.encode(render(row['prompt'], row['rejected'])).ids)), 'overlap': orientations['overlap'] * math.fsum((left * right for left, right in zip(prompt, difference))), 'answer_only': math.fsum((left * right for left, right in zip(difference, bow))), 'surface': math.fsum(((left - right) / scaling * weight for left, right, scaling, weight in zip(chosen_style, rejected_style, scale, style)))}
            for name, value in values.items():
                if not math.isfinite(value):
                    raise ValueError('nonfinite frozen baseline score')
                totals[name] += 1 if value > 0 else 0.5 if value == 0 else 0
        return {name: total / len(rows) for name, total in totals.items()}
    return scores
