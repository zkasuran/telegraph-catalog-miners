# Threat model

SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0

| Attack | Defence | Check |
|---|---|---|
| Duplicate Status keys | Reject duplicate JSON keys | duplicate-keys test |
| Oversized or deeply nested answer | 16 KiB text, 12 depth and 2048 node ceilings | parser-bound tests and random bytes |
| Wrong suffix, subdomain or type | Full normalized DNS-label identity, not substring matching | identity tests and mutants |
| DNSSEC or HTTP 200 substituted for RCODE | Only DNS Status drives lookup success | HTTP/DNSSEC control |
| NODATA or null MX treated as failure | Status 0 stays true | NODATA/null-MX tests |
| Timeout turned into false | No score without observed Status and unique Question | unavailable tests |
| Audit or validation used for fitting | Train-only integer baseline, validation-only neural selection and sealed candidate before audit | data tests and remote selection record |
| Model operations on laptop | Guard before neural imports | remote-guard test |
| Forged runtime identity | Compare observed foundation, tokenizer and adapter identities to frozen manifests | prepared remote runtime test, not yet executed |
| Prefixed echo passes only after normalization | Literal bare-hash echo comparisons plus actual node input-root expression | serving/wire contract tests |
| Report fills omitted neural operations with expected flags | Report assembly requires completed parser/engine results and strict unchanged quality gates | artificial report failure controls |
| Public forwarding key becomes general SSH access | Unchanged restricted common.lease_for binds one remote port, denied command and expiry | lease wire test; actual remote lease remains unverified until serving mode runs |

## Known limits

The raw DNS answer is untrusted self-report. Identity binding and Status consistency cannot distinguish a fabricated report from a real one. No DNSSEC proof, resolver trust chain or freshness authentication is provided. Unicode names require explicit ASCII presentation; URLs and whitespace-padded names are rejected rather than guessed. JSON-only coverage excludes free text and nested canonical miner reports. No universal DNS-retrieval quality claim follows from the fixture results.
