# Score source-bound smart contract findings

This pipeline connects a contract finding to complete source and a stated call. The source reference compiles 152 original Solidity contracts and checks the disputed call plus a legitimate control in an offline EVM. Neural work runs only in the bounded public GitHub workflow.

| Boundary | Source pairs | Grouped implementations |
|---|---:|---:|
| Training, unchanged from the first source build | 64 | 4 |
| Validation, unchanged | 24 | 2 |
| Fresh audit, excluded from candidate selection | 64 | 32 |

The audit adds mapping and structured storage, inherited authorization, asserted-authority mistakes, hash-based caller checks, selector/chain checks, integer casts, packing, compound arithmetic and accumulation. Parameter variants are not new independent groups. Source and reports are original AI-assisted work, not independent human preference annotations. Previous exposed audit designs are excluded.

## Run the source reference

From any checkout, run `bash verification/owners/smart_contract_audit/verify.sh`. An optional first argument selects the output directory. The gate installs integrity-locked non-neural compiler dependencies if needed, compiles every source and checks real interpreter effects. Wrong labels and invalid Solidity must fail.

No private checkout path, sibling training module, wallet or RPC is required. No model is imported by the local release gate. Length, source-answer overlap and immutable answer-only baselines are checked before training. Exact normalized cross-split source duplicates are zero. Maximum token-set Jaccard is 1.0: identical bags of tokens can still encode different execution paths. This is not a diverse real-world contract audit.

## Remote candidate and qualification

`evaluator-owner-smart_contract_audit.yml` has train and serve modes. Training uses the frozen three-epoch rank-8 recipe on the unchanged training set. Validation may select saved bytes. A fresh runner audits both frozen candidate and seed-matched untrained packages in the actual pinned Telegraph reference engine.

Each claimed response format, prose, JSON and report, requires 80% pair accuracy, 70% for both labels and both mechanisms, five points over untrained and ten points over the strongest shortcut. Whole source and answer must fit the authoritative 384-token render. An answer-only ablation must remain below the shortcut ceiling. A passing report binds the actual run, commit, package hashes, source manifest, engine, tokenizer and fixed finite score-bit vectors. No missing result becomes a pass.

Publication, dispatch and testnet registration use the installed automatic action adapter. It keeps credentials and signing outside the owner. Passing source checks do not authorize registration. The complete independent report, exact public delta and live serving run are hard gates. A node-confirmed matching registration is a separate result.

## Serving

Serving downloads only exact immutable public candidate/report bytes. It builds the same pinned executor, reproduces recorded vectors and creates an ephemeral restricted forwarding key on the runner. Only the public lease is uploaded. The lease binds candidate, runtime and source proof to the same combined report hash. Requests use literal bare hashes, actual token roots, bounded JSON and finite FP32 bits. No private key reaches the controller or owner.

## Limits and terms

Coverage is bounded setter authorization and arithmetic-call effects. The reference does not prove reentrancy safety, proxy state, arbitrary bytecode or universal factual truth. There is no human adjudication and pretrained-model exposure cannot be ruled out. Source, constructed reports and measured model results are separate artifacts.

The source build is verified. Neural accuracy and canonical qualification are not claimed by this README. See `DATA-SOURCES.md` and `SECURITY.md`. Our additions use LicenseRef-zkasuran-SAND-1.0 with unchanged upstream grants.
