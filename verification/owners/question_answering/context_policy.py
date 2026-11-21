# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Train-only constructed context controls, distinct from source human ranks."""
import hashlib
import json
import math
import re
from collections import Counter

from source_data import norm, render

VERSION = "prompt-conditional-ranking-v1"
NEUTRAL_WEIGHT = 0.25
MAX_TOKENS = 256
MIN_AUXILIARY = 512
STOP = frozenset(["about", "after", "also", "answer", "before", "being", "between", "could", "describe", "does", "explain", "from", "have", "help", "into", "just", "know", "make", "more", "most", "please", "question", "should", "some", "than", "that", "their", "them", "then", "there", "these", "they", "this", "those", "through", "using", "want", "what", "when", "where", "which", "while", "will", "with", "would", "write", "your"])


def words(text):
    return set(re.findall(r"[a-z0-9]{4,}", norm(text))) - STOP


def disjoint_prompt(row, donor, tokenize):
    """Lexical disjointness is a sampling rule, not factual evidence."""
    own = words(row["prompt"] + " " + row["chosen"] + " " + row["rejected"])
    topic = words(donor["prompt"])
    if row["group_id"] == donor["group_id"] or not own or not topic or own & topic:
        return False
    return all(len(tokenize(render(donor["prompt"], row[side]))) <= MAX_TOKENS
               for side in ("chosen", "rejected"))


def plan(rows, tokenize):
    if len(rows) != 768 or len({row["group_id"] for row in rows}) != 768:
        raise ValueError("context plan requires the exact independent training population")
    if any(row.get("source_split") != "train" or row.get("label_provenance") != "original human sibling rank"
           for row in rows):
        raise ValueError("only original public training source may construct auxiliary controls")
    usage, mappings = Counter(), []
    for index, row in enumerate(rows):
        order = sorted(range(len(rows)), key=lambda j: (
            usage[j], hashlib.sha256((VERSION + row["group_id"] + rows[j]["group_id"]).encode()).hexdigest()))
        for donor_index in order:
            if usage[donor_index] >= 2 or not disjoint_prompt(row, rows[donor_index], tokenize):
                continue
            mappings.append({"source_index": index, "foreign_prompt_index": donor_index,
                             "source_group_id": row["group_id"], "foreign_prompt_group_id": rows[donor_index]["group_id"]})
            usage[donor_index] += 1
            break
    if len(mappings) < MIN_AUXILIARY:
        raise ValueError("insufficient bounded disjoint training context controls")
    return {"schema_version": 1, "version": VERSION, "fit_split": "train",
            "source_preferences": len(rows), "auxiliary_controls": len(mappings),
            "source_labels": "unchanged original human sibling ranks",
            "auxiliary_labels": "constructed prompt-displacement neutrality assumption, not human-reviewed",
            "source_text_grant": "Apache-2.0", "additions_licence": "LicenseRef-zkasuran-SAND-1.0",
            "maximum_foreign_prompt_uses": max(usage.values()), "token_cap": MAX_TOKENS,
            "serving_remap": False, "mappings": mappings}


def plan_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def objective(own, foreign):
    if any(type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e6 for value in (own, foreign)):
        raise ValueError("finite bounded reward margins required")

    def softplus(value):
        return max(value, 0) + math.log1p(math.exp(-abs(value)))

    return softplus(-own) + softplus(-(own - foreign)) + NEUTRAL_WEIGHT * foreign * foreign


def derivatives(own, foreign):
    objective(own, foreign)

    def negative_sigmoid(value):
        e = math.exp(-abs(value))
        return e / (1 + e) if value >= 0 else 1 / (1 + e)

    bt = negative_sigmoid(own)
    conditional = negative_sigmoid(own - foreign)
    return -bt - conditional, conditional + 2 * NEUTRAL_WEIGHT * foreign
