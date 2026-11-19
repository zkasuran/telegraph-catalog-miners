# SPDX-License-Identifier: MIT
# Copyright (c) 2026 telegraphprotocol
import random
def split_pairs(pairs, eval_frac, seed):
    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)
    n_eval = max(1, int(len(pairs) * eval_frac)) if len(pairs) > 1 else 0
    ev = set(idx[:n_eval])
    return [pairs[i] for i in idx if i not in ev], [pairs[i] for i in idx if i in ev]
