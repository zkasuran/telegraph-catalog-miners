# Keep the exact scorer through a bounded lease

The original M2 package and audit stay unchanged. A lease generation carries the
byte-identical audit report and the previous public restricted lease. Its source
gate rejects changed model bytes, audit identity, executor source and generation.

`lease_continuation.py` builds the next serving dispatch from those artifacts.
The new report URL names the published lease generation that the original
executor will fetch and hash. This is an explicit consumed artifact, not a dummy
workflow input or a new model trial. The package URL remains unchanged.

The serving job still runs the original audited source commit. Its existing
workflow concurrency group has `cancel-in-progress: false`, so a queued next
lease cannot cancel or replace the healthy original runner. Once the next public
lease exists, activate it through the existing adapter and check actual public
reference vectors. A source descriptor is not an active endpoint or registration.

`lease_readiness.py` checks the immutable continuation receipt, the actual
workflow/commit/time run identity and the complete public-lease artifact digest.
Only the literal one-file `lease.json` ZIP is accepted. Activation payloads use
the original restricted-key/expiry contract and must match an active runner.
Readiness for activation is not proof that forwarding or registration is live.

## Limits

GitHub's concurrency queue can leave a gap after the original runner ends. A
new lease is usable only after its actual run, public artifact and reference
vectors pass the activation checks. Existing signed registrations must not be
submitted again. An unsigned request whose serving runner changes needs explicit
gate-preserving reconciliation or a verified integration rebind, never a blind
retry. No balance restoration, registration or canonical rank is claimed here.
