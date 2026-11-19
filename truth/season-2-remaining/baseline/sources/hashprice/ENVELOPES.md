<!-- SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0 -->
# Correlated response-envelope compatibility

The coordinator's deployed failure occurred on getblockhash with one subrequest,
no numeric RPC error code and no HTTP status. The old local guard required the
2.0 marker and numeric ID 1, then required a result field. It discarded HTTP
metadata on local failures. This revision reports the observed envelope shape
and HTTP status without disclosing foreign IDs, remote messages or error data.

## Primary rule [CORE-RPC-INTERFACE]

Bitcoin Core v29.0 documents JSON-RPC 1.1 versus 2.0. A 2.0 request has the exact
jsonrpc marker. The modern response has that marker and only one of result/error.
The legacy response has no marker and both result and error members. The server
was legacy-only through v27.0. JSON-RPC 1.0 and 2.0 specifications also require
the response ID to match the request [JSONRPC-1, JSONRPC-2].

The worker sends numeric request ID 1. Both accepted response shapes require
that exact numeric value. String "1", null, absent and foreign IDs are rejected.
Markerless replies require both fields, with error null for success or result
null for an error. A declared 1.0 marker, unknown marker, array response, missing
field or conflicting result/error is not accepted. Modern replies require the
exact 2.0 marker and result/error exclusivity. Only documented top-level members
are accepted. Errors require a numeric integer code and string message. Remote
message and data content never leaves the parser.

Each successful result must match its specific RPC method's basic shape. Header,
canonical identity, difficulty, coinbase, Merkle inclusion and reward checks remain
unchanged in verifyBundle and the proof modules. No request or retry policy changes
are made. Legacy support does not permit a latest-block substitute or cached guess.

## What was observed

Three Node and three local workerd canaries sent the same real getblockhash request
with ID 1 and params [0] to PublicNode. All six replies were HTTP 200, JSON-RPC 2.0
and numeric ID 1. Their safe standard response bytes and hashes are recorded in
evidence/envelope-canary.json. No legacy reply was observed in these canaries.
This does not measure the coordinator's deployed edge environment.

The deployed cause remains unproven. The fix for losing HTTP metadata is directly
tested. Documented legacy compatibility is regression-tested, not presented as
the observed cause. Three weakened-source mutations must fail the tests. Five
workerd controls exercise malformed, foreign-ID and unsupported-marker replies.

## Next deployed observation

Apply HASHPRICE-ENVELOPE.patch to the diagnostic worker at the before hash in
ENVELOPE-RESULT.json or integrate the exact new publication/worker.mjs. The
cumulative HASHPRICE-RELIABILITY.patch is for the earlier original baseline.
The coordinator owns deployment and shared-copy rebinding. This run does neither.

On a deployed failure, record envelope_jsonrpc, envelope_id_type, result_present,
error_present, top_level_type and actual http_status. envelope_id exposes only our
public ID 1 or its string lookalike; a foreign ID is represented by null. Use the
typed metadata to distinguish legacy compatibility from missing result, wrong ID
or a proxy-shaped response. Do not clear upload readiness until the new deployed
gate passes. Actual Workers Free CPU consumption and node admission are unverified.
