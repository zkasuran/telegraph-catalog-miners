# General preference preparer threat model

| Attack | Defence | Tested control |
|---|---|---|
| Changed source or grant | Pinned archive size/SHA-256 and card/grant SHA-256 | source hash and positive full scan |
| Validation data admitted as training | Literal partition check before any text access | malformed validation records skipped |
| Invalid ranking annotation | Finite range, exact three annotations, spread and direction checks | invalid/neutral/disagreement cases |
| Alias or shared-answer leakage | NFKC/case/space identities and connected components before allocation | union and overlap controls |
| Conflicting preferences | Exclude entire connected component | inverted preference control |
| Count padding or duplicate observations | One source original per group, strict counts | insufficient count refusal |
| Audit-based development | Prepare audit file without measuring it; train-only frozen baselines | fitting audit.jsonl refused |
| Unbounded input | File, decompression, line, count and time ceilings | oversized/symlink and malformed source checks |
| Replay destroys retained data | Existing output refused before input reads | early replay refusal |
| Floating-point refitting during reproduction | Immutable hexadecimal binary64 coefficients | identity/nonfinite/overwrite controls |

Known limits: grouping establishes normalized prompt/answer separation only.
It does not establish entity/time independence or factual source correctness.
Byte-length screening does not prove the node tokenizer retains the preference.
The source gate has no model imports and cannot create registration-ready proof.
Original source and audit rows must stay out of public manifests and uploads.
