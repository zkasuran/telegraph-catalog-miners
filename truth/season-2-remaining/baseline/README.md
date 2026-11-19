<!-- SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0 -->
# Three truth sources on the existing hashprice Worker

The dispatcher adds exact company status, pinned account balance and verified
Bitcoin hashprice paths to the existing `telegraph-s2-hashprice` Worker. All eight
copied modules match their current source bytes. Existing paths use the original handler.

This diagnostic candidate is locally prepared source. The earlier shared wrapper
is deployed, but this revised hashprice module is not. No signature, upload or
Telegraph admission is claimed. The paths are free evaluation infrastructure.
No new Worker, paid compute, upgrade or service resale is required.

## Diagnostic candidate

This revision changes only `sources/hashprice/worker.mjs` among the eight copied
runtime modules. The owner provides method-level safe RPC diagnostics and retries
only numeric code -28, Bitcoin Core warmup. There are at most three attempts per
method and four extra calls per request, under the existing 15-second deadline.
Missing data, internal errors, HTTP errors and unknown gateway codes are not retried.
Remote error messages and response data are not echoed.

Previous live failures remain recorded. A new deployed success/failure gate must
test heights 800000, 840000 and 900000 before reliability or upload readiness is
claimed. Local replay does not prove this patch fixes those earlier failures.
Current module and reference hashes are in PROVENANCE.json and MODULE-HASHES.json.
The owner's current reliability notes and tests are retained byte-for-byte.

The current delivery also accepts documented markerless legacy Bitcoin RPC
replies. Numeric ID 1 must match exactly, both result/error members must exist
and the unused member must be null. Modern 2.0 replies still require result/error
exclusivity. Unknown markers, foreign IDs, missing members and ambiguous replies
are rejected. Safe envelope diagnostics retain shape and HTTP status without
echoing foreign IDs or remote content. See sources/hashprice/ENVELOPES.md and
ENVELOPE-RESULT.json. This compatibility change is not claimed as the observed
cause of the prior deployed failure.

## Paths and unsigned material

The existing origin is `https://telegraph-s2-hashprice.margyn.workers.dev`.

| New path | Input | Scored body field | Material |
|---|---|---|---|
| `/truth/v1/company` | company, jurisdiction | active Boolean | truth/company.yaml and company.jsonl |
| `/truth/v1/balance` | address, block; optional chain, metric | balance_gwei number with exact six-decimal wire text | truth/balance.yaml and balance.jsonl |
| `/truth/v1/hashprice` | block_height | hashprice_sats_per_ph_day number with six-decimal wire text | truth/hashprice.yaml and hashprice.jsonl |

Descriptors use the existing origin plus `/truth/v1` as base_url. Endpoint paths
are `/company`, `/balance` and `/hashprice`. Only those descriptor routing fields
changed. Source kind, Intent and original row provenance remain intact. All
JSONL rows embed their exact licensed YAML and are unsigned candidates.

## Original service

`/health`, `/__last`, `/hashprice`, the literal block-height template path and
all original method behavior remain in the captured original module. Unknown
paths return its 404. Existing refusals remain HTTP 200 with null numerical
fields. New truth refusals use non-success HTTP and omit the scored field.

The wrapper forwards only a changed pathname to dedicated handlers. Host,
query, method and body remain intact. Original requests retain object identity,
env, execution context and handler receiver. No response header is rewritten.

Wrangler has bundling disabled and an explicit module rule list, so dry-run
output preserves each source module's bytes as well as the original import graph.
The name and compatibility date remain `telegraph-s2-hashprice` and `2025-06-01`.
Captured bindings are empty, keep_vars is true and observability is false.
No deployment route, account identifier or paid-tier option is added.

## Runtime adaptation

Cloudflare does not implement redirect:error. The corporate and onchain modules
expose injected-fetch hooks. `workers-fetch.mjs` uses those hooks to map error
to manual with explicit redirect rejection. Fixed upstream URL, request body,
method, headers and abort signal remain intact. No global fetch shim is installed
in the Worker. The original handler uses its own unchanged transport.

## Reproduction

Use Node 22 or later and existing Python packages PyYAML and jsonschema. The
portable gate checks source hashes, descriptor schema, the actual dispatch entry,
the current owner source/retry suites, original coinbase proofs and wrapper mutations:

```sh
bash verify.sh
```

For a credential-free Wrangler check with an existing pinned Wrangler 4.163.0:

```sh
wrangler deploy --dry-run --config wrangler.json --outdir .dry-run
```

That command only compiles and prepares module output. It does not deploy. The
full private coordinator gate also parses actual wire numbers with Decimal,
checks source-input hashes and drives dry-run modules in local workerd against
real public sources. It records live deployment separately.

## Trust and limits

Corporate truth is the GLEIF-reported entity status, not national good standing.
Onchain truth compares two service witnesses at a canonical block hash. It does
not prove consensus or an account trie. Hashprice truth verifies a mainnet header
and coinbase proof but has one runtime provider and no full-chain validation.
Unavailable blocks and unavailable services fail closed.

The original hashprice service accepts proof-verified actual coinbase output
sum, including underclaimed reward. The truth handler refuses underclaimed
reward. That difference stays intact because both source modules are preserved.
Original upstream data-republication rights remain unsettled. No original
upstream response archive is included. See DATA-SOURCES.md and SECURITY.md.

## Licence and rights

Source: LicenseRef-zkasuran-SAND-1.0. Full component licence texts, notices and
evaluation grants remain in their source directories. The package supplemental
evaluation permission covers all included licensor-authored components. Third-party
service terms still apply. GLEIF data is CC0-1.0. PublicNode access is for free
evaluation only. No competing miner or RPC resale grant is made.

AI assistance: OpenCode, agentrouter/gpt-6.1-sol. No human review is claimed.
