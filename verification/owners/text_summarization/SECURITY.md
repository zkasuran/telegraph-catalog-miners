# Security and evaluation boundaries

| Threat | Defence | Evidence path |
| --- | --- | --- |
| Source replaced or grant missing | Exact revision, XML/Git hashes, primary title/abstract/author/grant reproduction | `primary.py`, source negative controls |
| Shortened decisive context | Complete source fields, literal rendering and every format under 256 tokens | `data.py`, `gates.py`, changed-context controls |
| Source/entity/time leakage | Grouped author/ORCID/DOI identities, normalized overlap and ordered task availability | `data.py`, split mutation controls |
| Audit used for tuning | All nine fixed main epochs, validation loss selection and restored checkpoint before audit scoring | `training_policy.py`, `static_check.py` |
| Repeated bootstrap examples inflate data counts | Eight exact original pairs, four train groups, declared 64 repetitions per pair | `role_bootstrap.py`, source/exposure and admission tests |
| Held-out data substituted into warm-up | Exact frozen manifest and complete training-row equality, preserved titles/source bytes | `curriculum.py`, split/title/source mutation tests |
| Warm-up silently replaces main training | Fixed 64-update warm-up, fresh main optimizer and complete main epochs | `static_check.py`, actual-source mutation tests |
| Floating-point baseline refit drift | Immutable train-only coefficient bytes and identity checks | `baseline.py`, baseline mutation controls |
| Bad tensors or metadata | Full pinned upstream package parser with fresh live-class binding | `packagecheck.go`, actual remote parser |
| Wrong runnable identity | Literal bare lowercase hash echoes and exact batch input root | `wire.py`, wire negative controls |
| NaN, infinity or malformed score | Integer FP32 bits decoded and checked finite | `wire.py`, score negative controls |
| Invalid tokens or excessive request | Reference empty/negative-token checks, bounded serving front end | `runtime_check.py`, remote controls |
| Non-repeatable runtime | Same-process and fresh-process fixed vectors | `runtime_check.py`, actual remote bit streams |
| Local neural execution | Guarded remote entry points, source-only imports locally | Local entry-point refusal tests |
| Unbounded source or archive | Per-file byte/time limits, six source workers, bounded archive members and no links | `data.py`, `reference.py`, `primary.py` |
| Licence absent on failed model | Full source/model/own grants on every partial export | `preserve.py`, `licensing.py`, byte-preserving stamp tests |
| Confounded batch diagnostic | Identical initialization, learning rate, update count and per-row exposure, only grouping differs | `reciprocal.py`, schedule/source mutation tests |

## Known limits

Three fixed recipes failed quality. Training-only diagnostic mode uses
four existing training groups to test parameter updates and source-role binding.
It never uses validation/audit outcomes and cannot emit registration readiness.
Its tiny-group fit metrics are not independent capability evidence. Source and
failed-run grant preservation still applies.

The first real model completed the full source/parser/reference-engine pipeline
but failed quality near chance in all response formats. Its 64 scored audit
groups are excluded by `exposed-audit.json`. Revision 2 requires complete
development quality and untrained/shortcut gains before opening a fresh audit.
`audit_policy.py` and named negative tests enforce this boundary. Revision 3
also failed development at 33/64 in every format. Single-source tiny-train fit
did not establish two-source generalization. The next diagnostic fits only four
existing training groups with matched stochastic versus reciprocal batches.
It uses no validation/audit scores. Even successful tiny training fit would not
prove generalization or independence from overlap shortcuts.

The matched-batch run timed out after 320 stochastic updates. Its original-task
training fit reached 8/8, but no reciprocal result exists and the comparison is
incomplete. Revision 4 uses that training-only role-bootstrap path before all
nine main epochs. It does not fit either held-out split or lower a quality gate.
Tiny bootstrap fit is not independent quality. The new candidate remains
unmeasured until its real remote run and complete parser/runtime checks finish.

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

## Frozen passing package

The candidate passed all original audit/parser/engine gates but the final report
formatter rejected the nine-epoch result with a stale three-epoch predicate.
`frozen_candidate.py` checks original package and report bytes against their
immutable hashes. It checks original embedded grants, not a rewritten notice.
`remote.py` refuses retraining this audited candidate. Recovery runs no fitting
and scores no audit rows. It repeats actual fixed vectors and wrong identities
in the pinned reference engine, then combines those results with the original
audit evidence. Tests reject changed epoch counts, package bytes and freeze flags.
Original training and recovery identities remain distinct in the ready report.
This is integration recovery, not a new independent audit or canonical win.

`released_report.py` checks the exact observed report and manifest hashes plus
original/recovery/serving identities before publication. The report is not
rewritten for a later source publication. Serving uses its verified source commit
with immutable proof URLs and repeats its actual vectors before forwarding.
The released proof byte-tamper test rejects changed manifests. Live endpoint
activation and registration must still pass the adapter's independent gates.
