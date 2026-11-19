# Integration threat model

| Threat | Defence | Verification |
|---|---|---|
| New route shadows old service | Exact four-path allowlist; unchanged Request/env/context forwarding | Identity test plus 1000 lookalike paths |
| Duplicate or injected query keys | Original source allowlists and limits | Wrapper refusal controls |
| Missing source becomes zero | Original fail-closed handlers | Missing validator and failed upstream controls |
| Wrong units or decimal coercion | Independent Decimal extraction and APY calculation | 400 independent arithmetic vectors |
| Wrong validator epoch or identity | Exact index/public key and genesis-plus-epoch calculation | Stale, future and wrong-epoch controls |
| Forecast represented as observation | METAR-only source with raw/XML station/time agreement | Forecast and wrong-time controls |
| Redirect or oversized upstream | Original manual redirect refusal, response ceilings and timeouts | Redirect, size and hanging-call controls |
| Source changes after verification | Allowlist and SHA-256 manifests | Package hash gate before and after tests |
| Private evaluation rows leak | Explicit public copy inputs; no catalog or preference pairs | Descriptor-only JSONL and private-path scan |

## Known limits

The operations handler's responses retain its original headers. It has nosniff and
no-store but no CSP. This is unchanged source behaviour, not a new hardening claim.
Two validator reads can take up to 24 seconds. Finance has a 45-second total ceiling.
The wrapper does not alter either source's semantics or add retries.
Runtime proof is local workerd plus real upstream data. Remote deployment identity,
Workers Free CPU usage, native codec and source admission remain unverified.
The source owners' exact-historical Beacon verification was unavailable. Their
latest-finalized check is not a same-epoch witness. No latest result is relabelled
as an exact delayed epoch. Baseline source-rights limits remain in its dossier.
