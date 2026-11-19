# Source-bound smart audit threat model

| Attack | Defence | Test or limit |
|---|---|---|
| Incorrect generated vulnerability label | Every source compiles and its disputed call runs in an offline EVM | `failure_controls.py` flips a label and requires rejection |
| Fake compiler pass | Zero compiler diagnostics, exact returned values and stored effects | `execute.mjs` positive plus legitimate controls |
| Invalid or substituted source | SHA-256 binding and compiler rejection | Unit tamper check plus invalid-source runtime control |
| Training leakage | Whole designs and vulnerable/fixed siblings remain grouped | Normalized clone and group-leak negative tests |
| Annotation shortcut | No vulnerability annotation in source, both reports share finding words | Length, answer-only and overlap checks |
| Candidate trained on audit | Separate audit job receives already frozen package hashes; training bytes are unchanged | Run/commit/package bindings and automatic adapter recheck |
| Truncated guard or precondition | Reject any full render above 384 authoritative tokens | Remote tokenizer check, not run locally |
| Endpoint returns another model | Exact foundation, tokenizer, adapter and token-root echo | Pinned literal client test plus real remote audit |
| Floating baseline refit drift | Fixed integer coefficients, no fresh floating-point fit | Training identity and frozen-file SHA-256 |
| Local neural operation | Guard before any neural import or download | Local authority refusal test |
| Parser failure hidden by a wrapper | Persist actual stdout/stderr before checking exit status; a dedicated remote mode checks frozen packages without retraining | Diagnostic entry refuses local execution; remote real-parser result is still pending |
| Stale or guessed model classification | Exact live family/child URLs, source hashes, freshness and matching model classes before private loaded-class parser call | Wrong class, wrong identity and stale-source refusal controls; no compiled/shared catalog edit |
| Private or unrelated job repository | Exact owned identity and boolean public flag from the hosted dispatch payload; protected default environment context | Positive public control plus private, ambiguous and wrong-repository refusal controls |
| Missing or malformed collator input | Default only omitted margin to 0.0 at the training boundary; reject nonfinite/nonnumeric values without mutating source | `test_contracts.py` literal-text preservation and invalid-margin controls; primary CLI smoke |
| Resource exhaustion | 256 rows, 2048 source bytes, 8192 bytecode bytes, 1 million interpreter gas, bounded subprocess/HTTP timeouts and eight serving connections | Wrong source and control tests; serving refuses malformed tokens before inference |

## Known limits

This is not a production contract audit. It tests only self-contained setter authorization and bounded arithmetic effects. It does not establish reentrancy safety, complete exploitability, mainnet state or universal factual truth. High lexical overlap remains despite whole-design splits. Source and claims share AI-assisted authorship. There is no independent human adjudication.

No credentials, wallets, transaction objects, signing or RPC calls are used in local verification. The interpreter evaluates bytecode directly in disposable memory. Neural and serving jobs are bounded standard public CPU tests. The serving runner creates only its own ephemeral restricted forwarding key. Only its public lease is exported. Testnet signing and endpoint activation remain inside the installed automatic adapter. No canonical qualification is established by source checks alone.
