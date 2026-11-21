# Threat model

| Attack | Defense | Control |
|---|---|---|
| Non-string/malformed identifier | Strict type and checksum refusal, no false VAT value | non_string_candidate_ids and normalization tests |
| Negative/leading-zero/wrong source pointer | Canonical unsigned array pointer and bounds | noncanonical_source_array_pointers |
| Changed raw or projected facts | Exact hash and field projection comparison | raw_tamper_and_projection_tamper |
| Alias/group leakage | Current/historical names normalized, shared and declared groups excluded | alias_and_related_groups |
| Insufficient sources | Stop rather than pad | no_padding_when_insufficient |
| Body exhaustion or redirect | 2 MiB response, 20-second requests, 600-second collection, fixed hosts | payload_ceiling and redirect controls |
| Validation/audit fitting or refitting | Train-only artifact, exact hex coefficients and overwrite/hash refusal | baseline tests |
| False remote identity | Require actual repository, run, commit and attempt or label local | remote_identity_not_inferred |
| Paid/unbounded runner or privilege escalation | One ubuntu-24.04 job, ten-minute bound, read-only permissions | runner_and_timeout_refused and privilege_and_unpinned_action_refused |

Raw cache keys are constrained; symlinks and oversized files are refused. Raw
transport can contain unneeded corporate fields, so keep it private and exclude
it from uploads. Dataset projections omit contact details, roles and addresses.

## Known limits

The dated public registry observation is not universal vendor legitimacy or
independently corroborated tax truth. One acquisition day does not establish time
generalization. Undeclared corporate relationships remain unknown. A checksum is
not company existence. Display names do not determine VAT membership or certify
identity. Missing evidence is unknown, never VAT false. The source task is solved
by exact lookup and the overlap gate blocks neural work. Pinned token retention
and deployed trusted-context delivery remain unverified. No delta or live executor.

The separate preference/ experiment does not change the observation diagnostic
or its gates. Its grouping, sealed-audit boundary, immutable baseline and bounded
CPU/upload controls are documented in preference/SECURITY.md. It does not qualify
a Vendor registration.
