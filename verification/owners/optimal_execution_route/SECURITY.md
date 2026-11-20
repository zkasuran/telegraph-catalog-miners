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
| Artifact upload omits hidden source reports or leaks build files | Closed report/log/licence allowlist staged into non-hidden result directory, upload only after safe staging | Staging, unexpected payload, symlink, size and hidden-path workflow controls |
| Contradictory labels hidden by context metadata or pair reversal | Compare exact primary rendered bytes, canonicalize only pair orientation and report an optimistic error ceiling | Reversal, opposite-label, same-answer and literal whitespace controls in test_wire_information.py |
| Duplicate state inflated into independent diagnostic rows | Deduplicate normalized source-group/pair/label identity and explicitly leave independence unverified | Duplicate reversal and retained three-observation/two-state controls |
| Hostile diagnostic annotations exhaust source checker | Strict row keys, integer-only preference labels, UTF-8 and bounded rows/bytes | Schema, Boolean, surrogate and aggregate-ceiling controls |

All 25 original oracle/transport tests remain. New boundary checks do not delete or lower any original assertion. Input response size is 16 KiB, source responses are at most 2 MiB and RPC batches at most 32 methods. Network calls have a 20 second timeout. The release gate imports only passive source tools and Python standard modules, not a neural library.

Source artifacts carry retained LICENSE, NOTICE and upstream grants. Each staged file has a bounded size and SHA-256 manifest. Missing outputs are marked incomplete, never filled from expected values. Unknown files, compiler binaries, tools and caches are not staged. The uploader stays at its default hidden-file exclusion. A staging safety failure prevents upload.

Known limits: two unsplit Uniswap V2 paths, one RPC provider, no verified Merkle proof and no gas-used or realized-execution measurement. Base fee is context, not total gas cost. Slippage minimum is a configured protection floor. Primary handler plumbing uses a mock score and is explicitly not the reference engine. No neural package, model-ready audit, endpoint lease or registration is produced. The source gate cannot solve the missing live reference-state boundary.

Information diagnostics accept at most 4096 supplied rows, 16 KiB per text field and 2 MiB aggregate input text. Labels and source-group identities are annotations, not independently verified truth. Exact rendered-text collisions guarantee identical tokenizer inputs, but this source-only check does not tokenize or measure truncation collisions. Its ceiling is conditional on the supplied rows and is not a claim that contextual modelling is universally impossible.
