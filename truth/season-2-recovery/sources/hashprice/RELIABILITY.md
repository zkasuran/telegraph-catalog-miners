<!-- SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0 -->
# RPC reliability diagnostic patch

The prior deployed reference intermittently returned upstream_rpc_error without
the failing method or RPC code. Follow-up successful requests do not classify an
earlier failure. This revision is a diagnostic patch with narrow warmup recovery,
not an empirical certification that those deployed failures are fixed.

## Primary error classification [CORE-RPC-ERRORS, CORE-RAWTRANSACTION]

Bitcoin Core v29.0 defines `RPC_IN_WARMUP = -28` as `Client still warming up`.
Only a numeric -28 in a valid JSON-RPC 2.0 error envelope is retried. Three attempts
per logical method are allowed, with 200 ms then 400 ms waits. A shared request
budget allows four additional calls total, so the eight-call path cannot exceed
12 subrequests. The original 15-second whole-request deadline includes these waits.

`RPC_INTERNAL_ERROR = -32603` should only describe genuine bitcoind errors, such
as datadir corruption. It is not classified as transient. Missing data -5,
miscellaneous -1, invalid parameters -8, unsupported methods and unknown gateway
codes are not retried. HTTP errors, malformed envelopes and transport timeouts
also remain fail-closed. Rate limits are not bypassed.

The primary getrawtransaction implementation accepts an explicit block hash. It
can return -5 for a block hash not found or a transaction missing from that block,
and -1 for unavailable block data. The fact that another peer later succeeds is
not sufficient documentation to classify either error as transient here.

## Public-safe signals

The correlated-envelope follow-up is in ENVELOPES.md. It restores HTTP metadata
on local checks and adds documented legacy response support. The actual deployed
envelope mismatch remains unproven, so diagnostics do not certify upload readiness.

Errors carry the method, numeric rpc_error_code or null, code type, HTTP status,
attempt count and retry-exhaustion flag. Remote error messages, response data and
keys are never copied. Success and failure include actual subrequest/retry counts
plus retry slots reserved. A missing numeric code is reported, not invented.

Retries reuse the exact origin, method and serialized parameters. All proof checks
remain intact. The final canonical height lookup still runs after proof verification.
There is no latest-block substitute, result cache or alternate paid tier.

## Coordinator step

Apply HASHPRICE-RELIABILITY.patch to the previously deployed worker module or copy
publication/worker.mjs into the wrapper's corresponding source slot. The module
hashes are in RELIABILITY-RESULT.json. Do not replace the shared wrapper or legacy
routes. This run does not deploy the patch.

After deployment, probe the actual endpoint at heights 800000, 840000 and 900000.
On a failure capture error.method, error.rpc_error_code and error.rpc_error_code_type.
Preserve the original error and distinguish retry exhaustion from unknown failures.
Only a new deployed success/failure gate can clear upload readiness. Local Node,
workerd and arithmetic checks do not measure the Workers Free 10 ms CPU budget.
CPU usage and actual node admission remain separate unverified gates.
