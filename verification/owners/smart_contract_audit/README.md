# Score source-bound smart contract findings

This pipeline connects a contract finding to complete source and a stated call. The source reference compiles 232 original Solidity contracts and checks the disputed call plus a legitimate control in an offline EVM. Neural work runs only in the bounded public GitHub workflow.

| Boundary | Source pairs | Grouped implementations |
|---|---:|---:|
| Training, 96 retained originals plus 48 identifier variants | 144 | 12 |
| Validation, unchanged | 24 | 2 |
| Fresh audit, excluded from candidate selection | 64 | 32 |

The audit adds mapping and structured storage, inherited authorization, asserted-authority mistakes, hash-based caller checks, selector/chain checks, integer casts, packing, compound arithmetic and accumulation. Parameter variants are not new independent groups. Source and reports are original AI-assisted work, not independent human preference annotations. Previous exposed audit designs are excluded.

## Run the source reference

From any checkout, run `bash verification/owners/smart_contract_audit/verify.sh`. An optional first argument selects the output directory. The gate installs integrity-locked non-neural compiler dependencies if needed, compiles every source and checks real interpreter effects. Wrong labels and invalid Solidity must fail.

No private checkout path, sibling training module, wallet or RPC is required. No model is imported by the local release gate. Length, source-answer overlap and immutable answer-only baselines are checked before training. Exact normalized cross-split source duplicates are zero. Maximum token-set Jaccard is 1.0: identical bags of tokens can still encode different execution paths. This is not a diverse real-world contract audit.

## Remote candidate and qualification

`evaluator-owner-smart_contract_audit.yml` has train, serve and parser-diagnostic modes. Training uses the existing rank-8 recipe with at most 12 epochs, learning rate 1e-4 and gradient accumulation 4. Balanced formats create 432 training views of 144 sources and 72 validation views of 24 sources. Views do not add independent source groups. Validation selects saved bytes by the worst margin to each supported format, label and mechanism gate, not the average alone. A fresh runner audits both frozen candidate and seed-matched untrained packages in the actual pinned Telegraph reference engine.

The latest paired-batch candidate failed real validation in run 44686261525: prose 0.75, JSON 0.625 and report 0.541667. Actual DataLoader controls passed but training memorized. Training-identifiers-v3 adds 48 execution-checked variants only within the same 12 training implementations to test identifier generalization. All 184 previous rows and both held-out files remain exact. These variants are not new independent groups or human annotations. New train-only integer baselines and sampling coefficients use separate versioned files; older versions stay untouched. There are 216 sibling batches and 54 accumulated updates per epoch. The longer owned CPU timeout is explicit and remains below 350 minutes. New neural quality is unmeasured.

Training samples these views using frozen positive integer implementation weights. Four retained designs now have 60 views each and eight newer pathways have 24 each. Versioned weights 2 and 5 give each implementation expected probability 1/12. The sampler draws 432 views per epoch with replacement. Older weights 1 and 4 remain preserved with their original training identity. Repeated draws are not additional source rows, preferences or human annotations. Coefficients are fitted on training only and never refitted during reproduction. The upstream dataset gate still checks the unique corpus.

The batcher draws vulnerable/fixed sibling pairs instead of independent rows. Each minibatch has two source views with the same call facts, the same format and exactly reversed literal preferences. There are 216 possible pair batches across 432 views. Batch size two and accumulation four are retained, giving 54 optimizer steps per epoch on the expanded training set. Pairing is a training-side contrast against source-independent answer bias, not new independent evidence or a model-quality claim.

Each claimed response format, prose, JSON and report, requires 80% pair accuracy, 70% for both labels and both mechanisms, five points over untrained and ten points over the strongest shortcut. Whole source and answer must fit the authoritative 384-token render. An answer-only ablation must remain below the shortcut ceiling. A passing report binds the actual run, commit, package hashes, source manifest, engine, tokenizer and fixed finite score-bit vectors. No missing result becomes a pass.

Publication, dispatch and testnet registration use the installed automatic action adapter. It keeps credentials and signing outside the owner. Passing source checks do not authorize registration. The complete independent report, exact public delta and live serving run are hard gates. A node-confirmed matching registration is a separate result.

## Serving

Serving downloads only exact immutable public candidate/report bytes. It builds the same pinned executor, reproduces recorded vectors and creates an ephemeral restricted forwarding key on the runner. Only the public lease is uploaded. The lease binds candidate, runtime and source proof to the same combined report hash. Requests use literal bare hashes, actual token roots, bounded JSON and finite FP32 bits. No private key reaches the controller or owner.

## Limits and terms

Coverage is bounded setter authorization and arithmetic-call effects. The reference does not prove reentrancy safety, proxy state, arbitrary bytecode or universal factual truth. There is no human adjudication and pretrained-model exposure cannot be ruled out. Source, constructed reports and measured model results are separate artifacts.

Prior candidates failed actual validation and remain unqualified. Source expansion raised JSON validation to 0.833333 in run 44642554891 but prose was 0.666667 and report 0.791667. The selected epoch's average hid weak format/label/mechanism cells. The selector-only experiment changed validation checkpoint selection to the worst gate margin while retaining the reported average, training source, optimizer and every qualification gate. Its three source changes were hash-verified and left the pinned kit untouched. No audit input enters selection. All old source rows, held-out bytes and immutable integer coefficients remain. New source-to-validation token-set Jaccard reaches 0.933333 and source-to-audit reaches 1.0 despite zero normalized clones. Real exported-engine validation must pass before exposing the independent audit. No new model improvement or canonical qualification is claimed. See `DATA-SOURCES.md` and `SECURITY.md`. Our additions use LicenseRef-zkasuran-SAND-1.0 with unchanged upstream grants.

The validation-only selector ran correctly in run 44662866270 but still failed: prose 0.75, JSON 0.583333 and report 0.75. Parser and inert package identity checks passed; the audit stayed unused. The current improvement targets measured training-group imbalance rather than a new random seed. The isolated trainer adaptation now adds one sampling anchor to the three verified selection anchors. Data, source groups, optimizer settings, validation selector and quality gates remain unchanged. Actual Torch sampler controls run only on the public CPU runner before training. New model quality is unmeasured.

Implementation-balanced sampling passed actual 12000-draw/repeat/refusal controls in run 44672243024 but its candidate still failed validation: prose 0.541667, JSON 0.541667 and report 0.708333. The audit stayed unused. A later local summary briefly carried an incorrect artifact ID; that invalid binding was preserved and rejected by a new harvested-identity guard before any publication or registration. The corrected result binds artifact 14603166508. Current paired-batch source tests do not imply improved model quality. The actual DataLoader and exported-engine gates run remotely before any independent audit or registration.
