# Academic title scorer threat model

SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0

| Attack | Defence | Check |
|---|---|---|
| Wrong paper behind a valid identifier | Exact PMID and OpenAlex binding, DOI/title alias union | Metadata identity and transitive alias tests |
| First-page population claim | Full RELISH file inventory plus count-checked 100-ID metadata batches | Complete-page and missing-ID tests |
| Relevance label disagreement | Exclude every relation with conflicting original labels | Consensus test |
| Source repeats across splits | Identity components before construction, no repeated selected entity/title | Cross-original overlap test |
| AI review presented as human approval | Separate original annotations, constructed labels and explicit AI-assisted review fields | Review provenance test |
| Truncated label object | All chosen/rejected renderings must fit the released 256-token cap intact | Source-only tokenizer preparation |
| Floating-point control refit drift | Freeze integer train-count coefficients, replay exact rational overlap | Baseline unit tests |
| Prefixed or padded hash echo | Literal bare lowercase string equality | Four-identity mutation tests |
| Nonfinite or false score bits | Strict unsigned integer and finite FP32 decoding | Score-bit negative controls |
| Invalid tokens or oversized wire body | Explicit batch, token, vocabulary and byte ceilings | Boundary/property tests |
| Model work on a workstation | Remote guard before neural imports | No-model-import failure test |
| Unlicensed failed checkpoint | Embedded full upstream grants and SAND addition terms | Binary payload preservation test |
| Changed cached source | Hash/URL/length-bound receipt check | Cache tampering test |

## Known limits

Title-only similarity does not prove every full-paper judgment can be inferred
from a title. Explicit AI-assisted source visibility review preceded the freeze.
Partial labels, ambiguous bindings and conflicting annotations are excluded.
Metadata and human source judgments may contain errors. There is no live browsing
or universal factual, DOI or result-count verification claim.

The current build implements frozen source preparation, immutable controls,
literal wire checks, compiled static parser integration and a guarded remote
training stage. Independent audit, trained-package checking, reference-engine
execution and restricted serving are implemented but have not run remotely.
No registration-ready report exists before those actual measurements pass.
Unit tests do not claim neural runtime success. The source release gate never
loads a model. Neural work requires the public CPU workflow identity guard.
