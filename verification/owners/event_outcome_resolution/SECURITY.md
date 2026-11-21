# Threat model

| Attacker and input | Defence | Test |
|---|---|---|
| Miner emits duplicate keys or contradictory yes/bool | Strict JSON object and exact boolean | exact_boolean_and_ambiguity |
| Miner returns another event's answer | Literal question-key binding | literal_answer_identity |
| Caller inflates text or sends unsupported grammar | 256-byte question and 4096-byte answer ceilings | input_ceilings |
| Names use Unicode or whitespace variants | NFKC/casefold identity with whole-name membership | unicode_identity, no_not_substring |
| Missing result is interpreted as no | Unknown outcome independent of bool false | unknown_is_not_negative, boolean_false_is_not_no |
| Publication uses corrupted or re-edited source data | Original pages, receipts and merged-record agreement | verify.py exact rebuild |
| Split leaks a repeated person or event | Full inventory identity ownership before split, overlap checks | group_overlap_rejection |
| Endpoint returns decimal float or wrong identity | Literal uint32 FP32 and count/identity checks | runtime_wire_failure_controls_no_model |
| Script launches a model on laptop | Workflow/repository guard before neural imports | remote_guard_before_model_imports |

## Known limits

Evidence in an arbitrary caller's question is not authenticated. The reference
measures conditional consistency with that evidence. Only source-derived prompt
construction checks official lineage. The API itself disclaims completeness and
accuracy warranties. A result lookup is not a general factual proof engine.
No market-specific settlement policy, cancellation/recount logic or provider
transport outside the declared event string has been verified. Python's strict
parser is a diagnostic, not the actual FP32 node scorer. Prepared remote code has
syntax and guard tests only until the controller runs it. There is no serving
endpoint and no qualification claim.

## Report evidence and class preflight

Ready-report fields require the JSON object and numeric types their contract
names. A boolean is not a schema version or zero source overlap. Malformed
containers and nonstring hashes fail with ValueError, not leaked parser errors.
Synthetic controls in test_ready_shapes.py check the unchanged quality floors.

Training checks live class and routing before opening frozen rows or fetching
neural assets. A source mismatch stays a refusal. Its failure artifact contains
only class/routing identities and response hashes, never source examples or pool
rows. test_preflight.py tests this path and its ordering without neural imports.

## Parent-only source plan

parent_scope.py binds the exact parent identity, judgment class and literal v4
field slots. It does not reclassify GENERAL or claim its coverage. The material
gate binds the parent's active snapshot and contributor count, preserving the
five-contributor floor. Tests refuse child-status substitution, altered wire
indexes and fabricated metric types. Generated plans leave validation accuracy
unset and never claim model or registration readiness. Remote integration and
production caller qualification remain unverified.
