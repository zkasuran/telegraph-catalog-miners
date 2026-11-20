# Recover the selected product-report model without retraining

The original broad-development run trained and selected a 512-token delta but
the experiment wrapper still required 256 tokens. The primary parser supports
512. The recovery binds an explicit expected token ceiling and preserves every
upstream tensor/header check, rank constraint, identity and failure control.

This directory contains only our exact licensed validation-selected delta, safe
original development proof assets and a remote-only recovery entrypoint. It does
not use the last-epoch checkpoint or train/select another candidate. Original
data hashes and frozen train-fitted baseline coefficients are required.

The source gate runs all previous tests plus five recovery contracts. Actual
Go regression tests use tiny artificial parser fixtures, never a local model.
Only the bounded public CPU recover_broad workflow may run model operations.
It checks all selected tensor bytes against the inverse/re-exported PEFT state,
measures development validation and runs the real parser/reference executor.

The published delta is not registration-ready. No independent audit exists and
the actual deployed judgment caller remains unverified. Data/model/source rights
are retained in the bytes and parent notices. Our additions use SAND, subject to
the original Apache-2.0, MIT and public-domain/CC BY source grants.
