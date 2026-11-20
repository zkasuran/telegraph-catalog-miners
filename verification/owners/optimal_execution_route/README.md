<!-- SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0 -->
# Score one ETH swap quotes against pinned state

Use this integer oracle to check a USDC output against a separately trusted Ethereum block. It compares direct WETH/USDC and WETH/DAI/USDC quotes through three Uniswap V2 pools. It checks full route identity when a route is supplied. Without trusted state it abstains.

This is a source-only baseline. It is not a judgment adapter, global route optimizer or execution certificate. No neural model is loaded by its release gate.

## Run the source gate

Use Python 3.11 or 3.12 on Linux x86_64 and Go 1.25.7. Run `bash verification/owners/optimal_execution_route/verify.sh` from the public checkout. The script installs only hash-locked passive source tools when needed. Both interpreter wheel hashes are pinned to the same reviewed PyYAML 6.0.3 release. ROUTE_GO_BINARY may point to an existing Go executable. All paths derive from the standalone package.

The original 25 tests and 1000 seeded integer invariants are retained. The historical 18 router view quotes are regression fixtures, not new training or independent audit data. The primary Go question function and executor handler compile in source-only plumbing. The executor has an explicit mock scorer, not the foundation engine.

The workflow stages only bounded source reports, logs and retained licence files into a non-hidden result directory. Its manifest binds exact file hashes. Tools, compiler binaries and private working files are excluded. Incomplete source checkpoints are labelled incomplete rather than presented as ready results.

## What the score means

The oracle checks fee-inclusive gross USDC base units for exactly 1000000000000000000 wei ETH. Mainnet USDC has six decimals. Every pool has a fixed 3000 ppm fee in this bounded policy. Equal gross-output routes are ties. It does not rank transaction gas, split allocations, other venues or future proceeds.

Both block and policy SHA-256 values must be supplied to the CLI. See `python3 verification/owners/optimal_execution_route/oracle.py --help`. A matching native amount_out succeeds. A one-base-unit mutation exits 2. No reference state from the miner body is treated as trusted.

## Qualification boundary

The current live request has only pair=ETH/USDC. The model executor receives token lists. Its input_root is a token-batch identity, not a trusted blockchain-state root. The retained two-state witness requires opposite exact orderings for the same question and candidate bytes. Candidate lengths match.

`qualification_gate.py --require-ready` therefore exits 2 with an evidenced blocker. This is expected refusal, not a model-ready report. Only two normalized regression-state groups exist. Training and independent audit pairs are absent. The workflow mode is source_verify only and uploads source-run proof, never registration-ready.json.

## Check label information before fitting

Run `python3 verification/owners/optimal_execution_route/wire_information.py` to replay the existing diagnostic. Add `--require-unambiguous` to require no unavoidable literal-wire label errors. The retained three observations collapse to two normalized state rows. Their identical rendered pair has opposite labels, so the optimistic deterministic ceiling on this diagnostic is 1/2. It is not a model accuracy estimate or a claim about the hidden evaluation population.

The reusable `assess(rows)` function accepts strict diagnostic annotations with source_group, question, answer_a, answer_b and preferred (-1, 0 or 1). It keeps scoring bytes literal, handles pair reversal, reports answer-only ceilings and refuses oversized or invalid annotations. Metadata is not added to the scoring input. The source gate compiles the real primary question builder, compares the pinned renderer and executes both CLI success and expected refusal. No original observation becomes new training or audit data.

## Source and licence

Source pins, grants and inventory boundaries are in DATA-SOURCES.md and source-manifest.json. Retained Telegraph and Uniswap portions keep their MIT grants. New competitive additions use the reviewed SAND terms in LICENSE. SECURITY.md names each trust boundary. No protected pool row, private working file, credential or prior hidden audit is published.

AI assistance: agentrouter/gpt-6.1-sol. No human review or canonical win is claimed.
