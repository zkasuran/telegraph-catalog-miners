# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Training-only nuisance control, not a serving-time score remap."""
import math
import random

GROUP = 16
WEIGHT = 0.25


def direction(row):
    difference = len(row["chosen"]) - len(row["rejected"])
    if difference == 0:
        raise ValueError("balanced nuisance groups require non-tied source lengths")
    return 1 if difference > 0 else -1


def optimizer_groups(rows, seed, epoch):
    if not rows or len(rows) % GROUP:
        raise ValueError("source population must tile fixed optimizer groups")
    sides = {sign: [i for i, row in enumerate(rows) if direction(row) == sign] for sign in (-1, 1)}
    if len(sides[-1]) != len(sides[1]):
        raise ValueError("training source length directions must remain balanced")
    rng = random.Random(seed + 1009 * epoch)
    rng.shuffle(sides[-1])
    rng.shuffle(sides[1])
    groups = []
    for start in range(0, len(rows) // 2, GROUP // 2):
        group = sides[-1][start:start + GROUP // 2] + sides[1][start:start + GROUP // 2]
        rng.shuffle(group)
        groups.append(group)
    rng.shuffle(groups)
    if sorted(i for group in groups for i in group) != list(range(len(rows))):
        raise ValueError("optimizer ordering omitted or repeated a source original")
    return groups


def nuisance_moment(margins, signs):
    if (len(margins) != GROUP or len(signs) != GROUP or sum(signs) != 0
            or any(type(sign) is not int or sign not in (-1, 1) for sign in signs)
            or any(type(margin) not in (int, float) or not math.isfinite(margin) for margin in margins)):
        raise ValueError("invalid balanced train-only nuisance moment")
    moment = math.fsum(margin * sign for margin, sign in zip(margins, signs)) / GROUP
    # For J = mean(BT) + weight * M^2, the extra derivative per margin is
    # (2 * weight * M) * sign / GROUP. The detached first pass measures M.
    return moment, 2 * WEIGHT * moment


def reference_objective(margins, signs):
    moment, _ = nuisance_moment(margins, signs)
    bt = math.fsum(max(0.0, -margin) + math.log1p(math.exp(-abs(margin))) for margin in margins) / GROUP
    return bt + WEIGHT * moment * moment
