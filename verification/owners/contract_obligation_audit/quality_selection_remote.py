# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only adapter preserves training/evaluation while selecting gate slack."""
from contextlib import contextmanager
from pathlib import Path

from contracts import remote_only
from quality_selection import compile_selector


@contextmanager
def selector_adapter(trainer):
    remote_only()
    source = Path(trainer.__file__)
    if source.is_symlink() or not source.is_file() or source.stat().st_size > 128 << 10:
        raise ValueError("bounded regular primary trainer source required")
    code = compile_selector(source.read_bytes())
    namespace = {}
    # Defines only the original train function with one metric expression changed.
    # Neural imports and calls still happen solely when the guarded trainer runs.
    exec(code, trainer.__dict__, namespace)
    original = trainer.train
    trainer.train = namespace["train"]
    try:
        yield
    finally:
        trainer.train = original
