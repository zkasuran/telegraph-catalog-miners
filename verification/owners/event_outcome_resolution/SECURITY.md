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
