# Threat model

SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0

| Attack | Defence | Check |
|---|---|---|
| Answer authenticates its own invented report | Only caller report is the source authority | Report removal and contradictory-report controls |
| Source belongs to another document | Literal URL and digest binding | Wrong document, SHA and report type tests |
| JSON ambiguity or oversized input | Duplicate keys, nonfinite values and byte ceilings refused | Parser adversarial and bounded random tests |
| Cross-split source reuse | Whole original/session groups and normalized prompt/answer checks | Frozen data gate |
| Overlap or length shortcut | Immutable train-fitted baseline and unchanged 65 percent gate | Known v1 refusal control |
| Local neural work | Guard before every remote path and before neural imports | Driver refusal tests and publication import guard |
| Modified package/runtime | Full hash checks, Stage 1 parser and real reference executor | Actual remote gates, not a local Torch stand-in |
| Hash alias or prefixed echo | Literal bare lowercase wire identities | Prefix/case mutation tests and remote identity refusals |
| Nonfinite or extra score result | Exact result count and finite FP32 bit validation | Score-bit and result shape controls |
| Failed candidate loses grants | Tensor-preserving stamp in finally before artifact upload | Stamp test plus actual remote checkpoint receipt |
| Forwarding key becomes a shell key | Reviewed common.lease_for restrictions and a pinned host key | Lease contract tests; automatic activation checks |

Bounded public CPU jobs are the only neural runtime. No local models, wallet
keys, provider credentials or private SSH keys are inputs. Source text and source
annotations stay distinct from AI-assisted preference construction.

Known limits: trustworthy production report delivery, signing-time issuer trust,
revocation and LTV remain unverified. One issuer and semantic clones may reduce
effective source independence. The source-held-out audit does not establish a
canonical win or guarantee performance on arbitrary evidence-free descriptions.
