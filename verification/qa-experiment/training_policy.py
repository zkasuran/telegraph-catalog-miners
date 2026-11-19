# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Fixed development-only epoch selection and isolated evaluation RNG."""
import math
import random
from contextlib import contextmanager

EPOCHS = 3


def development_selection(best, epoch, metrics):
    if type(epoch) is not int or not 1 <= epoch <= EPOCHS:
        raise ValueError("epoch outside the fixed three-epoch recipe")
    loss = metrics.get("val_loss")
    if type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0:
        raise ValueError("development loss must be finite and nonnegative")
    if best is not None and epoch <= best["epoch"]:
        raise ValueError("development epochs must advance")
    # Exact ties keep the earlier epoch. Accuracy and every audit metric are
    # intentionally absent from this decision.
    improved = best is None or loss < best["val_loss"]
    return ({"epoch": epoch, "val_loss": loss} if improved else best), improved


@contextmanager
def isolated_evaluation(model, torch, numpy):
    python_state = random.getstate()
    numpy_state = numpy.random.get_state()
    training = model.training
    # Development DataLoader construction can draw from CPU Torch RNG even
    # without shuffle. The primary fork_rng context restores that state too.
    with torch.random.fork_rng(devices=[], enabled=True):
        try:
            model.eval()
            yield
        finally:
            random.setstate(python_state)
            numpy.random.set_state(numpy_state)
            model.train(training)
