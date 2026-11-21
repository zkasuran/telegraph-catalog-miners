# Threat model

| Threat | Defence | Test or measurement |
|---|---|---|
| Wrong checkpoint or source data | Immutable SHA-256 pins and exact original run/commit bindings | Local asset and source tests |
| Fabricated preference or human review | Unchanged attributed public originals, AI review disclosure | Source/grant checks |
| Exposed audit used for tuning | No training function or optimizer in recovery; package bytes fixed | Source tests and replay report |
| Missing extracted helper globals | json supplied to exact SHA-verified function AST | Literal score smoke with non-neural stub |
| Wrong wire identity or nonfinite bits | Exact bare-hash request/reply checks and finite FP32 decoding | Negative identity and score-bit controls |
| Lease archive contract mismatch | Exactly one `lease.json` public artifact with unchanged restricted lease fields | Literal producer/workflow filename and canonical lease smoke |
| Parser or engine mismatch | Full pinned node parser and real foundation engine | Remote-only parser and runtime reports |
| Unbounded runtime | Standard public CPU, explicit job/download/request time and byte ceilings | Workflow and remote runtime |

Local release verification never imports neural libraries. Remote results are
not assumed: registration-ready.json is written only after actual complete
parser, format quality, repetition, fresh-process and identity checks pass.
No signer/provider key is handled by this pipeline. The automatic action adapter
owns publication and signing. Unknown action writes must be reconciled, not resent.

Limits: source-held-out quality is not a canonical node win. The fixed source
audit is not fresh after exposure. Engine performance and numerical behaviour
remain unverified until the bounded public job completes. No claim of human review.
