# Source and runtime threat model

| Attacker or failure | Defence | Verification |
|---|---|---|
| Source substitution | Immutable archive, split, policy and licence hashes | Source gate and cold archive test |
| Multiple or time-ambiguous values | Complete alternatives, precise qualifier eligibility and fail-closed source selection | Source fixture tests and primary-byte recomputation |
| False claimed source truth | Correctness is scoped only to actual pinned statements | Explicit source scope; no claim of universal external verification |
| Audit leakage | Frozen entity/relation splits, normalized overlap checks and dev-only checkpoint selection | Source gate and selection tests |
| Answer style shortcut | Immutable train-fitted length, lexical and answer-only controls | Local development controls; audit measured once remotely |
| Local/private/paid compute | Manual public repository workflow, hosted CPU guards and timeouts | Isolated environment negative tests |
| Parser or identity mismatch | Full current Stage 1 parser and literal bare-hash wire checks | Parser mutation tests and actual remote negative identities |
| Runtime drift | Complete audit vectors, repeated bits and fresh process | Actual reference-engine checks, not native score substitution |
| Untrusted runnable request | Token-only schema, byte/time/header limits and finite score checks | Public wire unit tests and serving preflight |
| Forwarding key abuse | Ephemeral restricted runner key, remote-only forwarding and bounded lease | Same public lease contract; no private key artifact |
| Failed audit reused for development | Successor reads only exact train/validation bytes; runner removes audit read permission before fitting | Source smoke denies audit reads and source gate checks the remote permission refusal |
| First page mistaken for a corpus | Full monotonic keyset inventory compared with primary aggregate counts | Truncated-population, duplicate-ID and query-injection controls |

The source release gate never imports a neural framework or loads a model. It
recomputes the source bindings but does not score the audit with a model.
Remote reports bind actual job, source commit, data, package, runtime and node
identity. Success on this source task is not canonical node qualification.

Known limits: the deployed judgment caller's trusted observation delivery is
unverified. The source population is a bounded ordered subset and the claimed
format is concise prose only. Unverified and the other graded factual verdicts
need separate reviewed coverage. Runner leases are finite and need continuity
through the exact approved executor; no paid fallback is configured.

The successor development mode does not emit a registration-ready artifact.
Its validation is exposed for model selection. A new independent audit is still
required. Wikidata query transport may be unavailable or rate-limited; inventory
requests use at least 65-second spacing and a fixed request budget. Separate
queries are not an atomic source snapshot.
