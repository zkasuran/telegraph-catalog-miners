# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Real Torch RNG controls, executed only by the remote workflow."""
from core import isolated_rng, remote_only


def controls():
    remote_only()
    import random

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    torch.set_num_threads(4)
    def fixture(mode):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        parameter = torch.nn.Parameter(torch.randn(4))
        optimizer = torch.optim.AdamW([parameter], lr=0.01)
        loader = DataLoader(list(range(8)), batch_size=2, shuffle=True)
        original_eval = DataLoader(list(range(4)), batch_size=2, shuffle=False)
        for _ in range(3):
            for batch in loader:
                optimizer.zero_grad()
                mask = torch.nn.functional.dropout(torch.ones(4), p=0.1, training=True)
                (parameter * mask * (batch.float().mean() + 1)).square().mean().backward()
                optimizer.step()
            list(original_eval)  # Original validation RNG consumption stays intact.
            def extra():
                list(DataLoader(list(range(4)), batch_size=2, shuffle=False))
                random.random()
                np.random.random()
            if mode == "guarded":
                with isolated_rng(torch, np):
                    extra()
            elif mode == "unguarded":
                extra()
        return parameter.detach().clone(), torch.get_rng_state().clone()
    plain, plain_rng = fixture("plain")
    guarded, guarded_rng = fixture("guarded")
    unguarded, unguarded_rng = fixture("unguarded")
    if not torch.equal(plain, guarded) or not torch.equal(plain_rng, guarded_rng):
        raise RuntimeError("guarded tracing perturbed real fixed-step parameter bytes")
    if torch.equal(plain, unguarded) or torch.equal(plain_rng, unguarded_rng):
        raise RuntimeError("unguarded DataLoader failure control did not discriminate")
    return {"kind": "real remote CPU Torch fixture, no foundation model",
            "guarded_parameter_bytes_equal": True, "guarded_rng_equal": True,
            "unguarded_parameter_bytes_differ": True, "unguarded_rng_differs": True}
