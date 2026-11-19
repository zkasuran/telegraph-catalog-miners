<!-- SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0 -->
# Trust and failure boundaries

| Threat | Defence | Proof |
|---|---|---|
| Plausible output substitutes source truth | Abstain without separately trusted state | Missing-state and miner-supplied-reference tests |
| Correct amount hides fabricated route | Bind chain, block, amount, token/pool order and fees | Complete route mutation controls |
| Quote changed by wrong input or unit | Fixed one ETH and native integral USDC base units | Wrong-size, decimal, Boolean and unit controls |
| Tied best route falsely rejected | Accept every exact gross-output tie in the bounded universe | Constructed tied-state test |
| Parser exception on hostile input | Size ceilings, duplicate-field refusal, DecimalException and UnicodeError boundary | Deep JSON, extreme exponent and surrogate controls |
| Changed reference bytes | Separately supplied SHA-256 pin and exact router replay | Hash tamper and mutated-state controls |
| Repeated state disguised as new example | Normalize reserve/pool identity independent of block number | Duplicate-state regression |
| Network write hidden as source call | Only five read-only RPC methods on one fixed endpoint | Transaction and signer methods refused |
| Model wire used as state oracle | Execute unchanged primary handler with a labelled token-only spy | Equal token calls despite distinct fake reference fields |
| Unbounded or paid workflow | Single ubuntu-24.04 job, 20 minute ceiling, pinned actions and no cancellation | Parsed workflow negative controls |
| Source gate mistaken for qualification | Explicit failing qualification readiness check | --require-ready exits 2 |

All 25 original oracle/transport tests remain. New boundary checks do not delete or lower any original assertion. Input response size is 16 KiB, source responses are at most 2 MiB and RPC batches at most 32 methods. Network calls have a 20 second timeout. The release gate imports only passive source tools and Python standard modules, not a neural library.

Known limits: two unsplit Uniswap V2 paths, one RPC provider, no verified Merkle proof and no gas-used or realized-execution measurement. Base fee is context, not total gas cost. Slippage minimum is a configured protection floor. Primary handler plumbing uses a mock score and is explicitly not the reference engine. No neural package, model-ready audit, endpoint lease or registration is produced. The source gate cannot solve the missing live reference-state boundary.
