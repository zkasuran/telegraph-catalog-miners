<!-- SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0 -->
# Shared dispatcher security boundary

The wrapper selects only three exact new paths. Each source keeps its parser,
upstream allowlist, timeouts and response bounds. No env-selected upstream exists.
Every other request goes directly to the original handler with its original
Request, env and execution context. Response bodies and headers are untouched.

The corporate and onchain injection hooks receive a scoped Workers transport.
Workers does not implement redirect:error, so this transport uses manual and
rejects redirect responses before their destinations can be called. Methods,
bodies, headers, signals and fixed upstream URLs are retained. The original
Worker never uses this transport. See [CF-REQUEST] in NOTICE.

| Input or attacker | Defence | Verification |
|---|---|---|
| Caller tries to change upstream host | Request host stays intact but handlers use fixed service URLs | Actual entry transport tests and host mutation control |
| Query or body mutation in forwarding | Only pathname is assigned; Request constructor forwards metadata and stream | Duplicate query, binary body, stream and cancellation checks |
| Existing route regression | Exact fallthrough to original.fetch | Original health, recent, hashprice, literal, OPTIONS, POST and 404 checks |
| Invalid corporate match or incomplete data | Exact identity, jurisdiction, complete collection and fresh publication | Copied corporate suite and actual-entry active/inactive/ambiguity tests |
| Wrong chain, block or account state | Two service witnesses, EIP-1898 hash pin and header recheck | Copied onchain suite and exact wire-number tests on three chains |
| Forged Bitcoin block or reward | Header SHA256d, target, coinbase ID, inclusion and reward checks | Copied hashprice suite, original coinbase checks and actual-entry refusal |
| Source bundle drift | Eight pinned hashes and explicit publication allowlist | Hash check before and after gate |
| Unlimited source reads | Existing per-source bounds and request limits retained | Copied timeout, oversized response and saturation tests |

## Known limits

The original routes intentionally retain original semantics and headers, including
HTTP 200 with null scored values on refusal. Truth routes refuse with non-success
HTTP and omit the scored field. No claim is made that original behavior is hardened
beyond its captured implementation. Original mempool.space and Blockstream data
republication rights remain unsettled. See DATA-SOURCES.md.

Onchain witnesses are service agreement, not a consensus or account trie proof.
Hashprice truth has one runtime provider. Full Bitcoin chain validation is not
performed. GLEIF is incomplete coverage and publication age is not entity age.
The original Worker and new truth handlers retain different hashprice definitions
for underclaimed reward. Neither source module is changed to reconcile them.

This is local verified source, not a deployed wrapper. New public routes, current
node loading, signature, admission and snapshot inclusion need coordinator proof.
The existing account remains Workers Free. There is no new Worker name, binding,
route declaration, paid-tier flag, builder or observability upgrade in the config.
