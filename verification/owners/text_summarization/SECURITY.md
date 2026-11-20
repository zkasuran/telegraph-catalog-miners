# Security and evaluation boundaries

| Threat | Defence | Evidence path |
| --- | --- | --- |
| Source replaced or grant missing | Exact revision, XML/Git hashes, primary title/abstract/author/grant reproduction | `primary.py`, source negative controls |
| Shortened decisive context | Complete source fields, literal rendering and every format under 256 tokens | `data.py`, `gates.py`, changed-context controls |
| Source/entity/time leakage | Grouped author/ORCID/DOI identities, normalized overlap and ordered task availability | `data.py`, split mutation controls |
| Audit used for tuning | All three epochs, validation loss selection and restored checkpoint before audit scoring | `training_policy.py`, `static_check.py` |
| Floating-point baseline refit drift | Immutable train-only coefficient bytes and identity checks | `baseline.py`, baseline mutation controls |
| Bad tensors or metadata | Full pinned upstream package parser with fresh live-class binding | `packagecheck.go`, actual remote parser |
| Wrong runnable identity | Literal bare lowercase hash echoes and exact batch input root | `wire.py`, wire negative controls |
| NaN, infinity or malformed score | Integer FP32 bits decoded and checked finite | `wire.py`, score negative controls |
| Invalid tokens or excessive request | Reference empty/negative-token checks, bounded serving front end | `runtime_check.py`, remote controls |
| Non-repeatable runtime | Same-process and fresh-process fixed vectors | `runtime_check.py`, actual remote bit streams |
| Local neural execution | Guarded remote entry points, source-only imports locally | Local entry-point refusal tests |
| Unbounded source or archive | Per-file byte/time limits, six source workers, bounded archive members and no links | `data.py`, `reference.py`, `primary.py` |
| Licence absent on failed model | Full source/model/own grants on every partial export | `preserve.py`, `licensing.py`, byte-preserving stamp tests |

## Known limits

The two fixed recipes failed development. Training-only diagnostic mode uses
four existing training groups to test parameter updates and source-role binding.
It never uses validation/audit outcomes and cannot emit registration readiness.
Its tiny-group fit metrics are not independent capability evidence. Source and
failed-run grant preservation still applies.

The first real model completed the full source/parser/reference-engine pipeline
but failed quality near chance in all response formats. Its 64 scored audit
groups are excluded by `exposed-audit.json`. Revision 2 requires complete
development quality and untrained/shortcut gains before opening a fresh audit.
`audit_policy.py` and named negative tests enforce this boundary. The new
optimizer recipe is not a measured improvement until its remote run completes.

This is a target-study title relevance experiment, not a general factual summary
judge. Source grants do not establish independent scientific truth. Preferences
are constructed, not human-reviewed. These grouped source audit rows are not the
protected node evaluation pool. They cannot be used to tune a later candidate
after model scoring exposes them.

The upstream answer-only surface diagnostic uses an internal pair-level split.
Its low accuracy is not a grouped-held-out quality result. Frozen controls are
reported on actual source-held-out splits and do not replace neural assessment.

The reference engine truncates overlength token input by design. Local whole-input
checks reject context beyond 256 tokens before scoring. Empty and negative token
refusals are checked against the real engine. Serving adds a bounded front end.
Python/native score agreement uses recorded numerical tolerances. Only repeated
reference-engine bits earn an exact-repeat claim.

Remote parser, training, engine and endpoint results are unverified until their
real jobs complete. A source gate, registration receipt or audit pass is not a
canonical node win. No local model loading, credentials or paid resources are
part of the release gate.
