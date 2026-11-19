# Comparator recovery security scope

This package is an offline prospective release and pure rollout planner.
No deployment, authenticated inventory, chain preflight or transaction was
performed. The supported runtime contract is sequential: allocate 16 bytes,
write two little-endian int64 values, call truth first and free the buffer.

| Surface or attacker | Defence and check | Limit |
|---|---|---|
| Changed arithmetic or build toolchain | `build.py` checks compiler commit, linked archives and every non-custom section against the historical base, allowing only the fixed-width tag replacement; `verify.py` runs real reference cases | Reference compatibility is not mathematical distance correctness over all int64 pairs |
| Missing portable build input or substituted validation source | Bundled original/family source, complete licences, pinned validation helpers and vendored dependencies; every executable hash/version and validation file hash is checked | Requires the explicitly pinned installed Linux AMD64 toolchains. No tool download occurs |
| Operator path leaked in source or binary | Gate scans every public file as bytes before and after reproduction | Generated scratch/evidence logs are excluded from publication |
| Missing linked notices or false family identity | Full pinned text comparisons and ordered custom-section inspection in `verify.py` | Publication commit is assigned by controller outside immutable binaries |
| Int64 wire rounding | Existing lossless lexeme parser, 690 ordered Worker cases and precision failure control | Full child extraction/scaling may still use float64 upstream |
| Invalid JSON, wrong types and invalid spec | `test_wire.mjs` and `test_resources.mjs` exercise actual Request/Response handlers | Numeric strings remain accepted by the reused handler |
| Oversized or slow request | Handler rejects above 1 MiB, caps batch at 100 and times out body reads at five seconds | Numeric parsing and regex CPU at the full body cap are not proven within Cloudflare's free-plan budget |
| Arbitrary WASM pointer or allocator misuse | 4097-byte allocation refusal and out-of-memory-end spec trap tested in Go | Original unsafe pointer API is unchanged; direct caller misuse can trap. Multiple outstanding allocations are unsupported |
| Excess memory or unexpected imports | No imports, exact export set and 1114112-byte memory per instantiated family tested | Cloudflare isolate aggregate memory/startup remain unmeasured |
| Replacement of old module, binding or route | `assertPreserved`, snapshot fingerprints and 19 negative planner controls | Snapshot completeness, actual entry exports and provider API normalization need controller verification |
| Concurrent independent deployer | Complete snapshot drift abort before promotion; rollback requires promoted fingerprint | No provider-side conditional write is implemented here; controller must serialize its operation and re-read after writes |
| Rollback removes registered URL | Compatibility repair retains every old module plus registered binary and runnable paths; deletion control fails | Dispatch repair cannot resolve a defective immutable WASM. It cannot undo a registry write |
| Receipt from wrong deployment or ambiguous broadcast | Prototype rejects old Diamond, wrong token, owner, nonce, calldata, hash, event/getter, stale proof and unresolved outcome in 24 negative controls | Input records are already decoded. ABI/code binding, signing, persistence and actual chain reconciliation are outside this prototype |

## Reference limitations retained deliberately

Signed subtraction wraps. Some extremely distant values can score as near
matches. MinInt64's wrapping absolute value leads to a denominator of one.
Very large tolerances can return one even for equal values. RelTol with zero
tolerance can score a sub-basis-point error as equal. Zero truth is refused
by the protocol host for all codes. Some wrapped results outside 0 through
10000 are refused; that refusal does not catch every misleading in-range score.

No arithmetic was changed to resolve those upstream rules. Twenty-five explicit
goldens, a signed boundary grid and full-range randomized raw/reference checks
make the scope visible. The node's deployed executable revision and full child
domain behavior still require controller evidence.

## Operations excluded from this package

There is no automatic publication, deployment, registration, credential loading,
model execution, service start, fund transfer or retake loop. The controller
owns the authenticated readbacks, release proof, rollback promotion and signing.
