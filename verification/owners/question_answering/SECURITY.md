# QA evaluator threat model

| Threat | Defence | Evidence |
|---|---|---|
| Source substitution | Exact revision and SHA-256 for both complete source files, source licence and tokenizer | Source preparation identity tests |
| Exposed audit reuse | Complete component and normalized-field exclusion, all prior 224 source originals excluded | Exposure manifest and source split checks |
| Shortcut learning | Unchanged kit gate plus frozen train-only length, token, overlap, answer-only and surface controls | Development gate, no bypass |
| Cropped labels | Both complete original renderings fit 256 tokens and retain different IDs | Token visibility gate |
| Audit-driven selection | Source design freezes before audit creation; three epochs select only development loss | Freeze checks, development-only policy tests |
| Local neural execution | Event/repository/temp-root guards; source gate has no neural imports | CI-environment refusal test |
| Package or identity swap | Full primary Stage 1 parser and literal bare wire identities | Parser and actual remote negative controls |
| Forged runtime agreement | Pinned engine/foundation, all audit bits repeated and fresh-process vectors | Actual reference executor report |
| Unlicensed checkpoints | Full own SAND plus separate upstream grants in byte metadata and sidecars | Metadata source tests and guarded remote roundtrip |
| Forwarding-key misuse | Ephemeral runner key, standard restricted public lease, exact report hashes | Lease contract; only lease is uploaded |

Known limits: source preferences are subjective and do not prove factual truth.
Source PII annotations are not an independent privacy audit. Pretraining overlap
is unknown. Source-held-out results are not canonical node qualification. This
code does not receive provider or signer credentials. Native/WASM comparison is
numerical and ordering agreement, not an assumed bit identity across runtimes.
