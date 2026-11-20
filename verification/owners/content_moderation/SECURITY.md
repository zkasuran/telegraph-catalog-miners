# Threat model

SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0

| Attack | Defence | Test or remote control |
|---|---|---|
| Local neural execution | Closed owned-workflow guard before neural entry points | Local guard tests |
| Audit-driven tuning | Trainer read denial, frozen selection, exclusive audit marker | Fake read-denial and freeze controls, remote marker |
| Source or model substitution | Immutable source hashes, package SHA-256, class snapshot, full parser | Local hash controls and real remote parser |
| Tracing perturbs training | CPU RNG fork plus Python/NumPy restoration | Fake local controls and real remote positive/negative fixture |
| Weak class hidden by aggregate | Every format and label required | Synthetic quality failure controls |
| Shortcut scoring | Train-only fitted coefficients, frozen byte hash, strongest-rule comparison | Frozen-coefficient controls and remote results |
| Wrong literal wire identity | Bare-hash output equality, actual reference bridge | Fake HTTP smoke and real remote controls |
| Nonfinite or missing scores | Strict finite FP32 and count checks | NaN, infinity, missing and duplicate controls |
| Untrusted remote time/size | Phase/process-group ceilings, body limits and source hash ceilings | Source gate plus remote phase logs |
| Credential or endpoint takeover | Only existing proxy, ephemeral restricted runner key, public lease only | Lease contract validation by automatic adapter |
| Fitted checkpoint presented as untrained | Separate immutable warm-start and genuine untrained asset hashes | Explicit source gate and distinction failure controls |

F3 uses a fresh low-LR optimizer from a validation-selected H9 snapshot. It is
not exact optimizer/RNG continuation. The H9 failure is not relabelled passing.
Source-only checkpoint term checks verify original grants in both containers
before publication. Successful registration evidence is separate from large
epoch diagnostics; the adapter's archive ceiling is not weakened.

No hidden-pool or development-exposed audit input is permitted. No neural model
is loaded locally. Source publication alone is not qualification. The runtime
bridge returns the identical engine's score bits, not a Torch substitute.

Known limits: one seed, narrow English extreme-toxicity proxy, unknown foundation
pretraining overlap and same-source held-out audit. Source-only tests do not
verify neural quality. Go compilation and actual reference-engine execution are
mandatory remote gates. Five-hour free serving leases require byte-preserving
renewal, not a new registration lottery. No paid infrastructure is created.
