# Keyword centrality pilot

This module prepares source-bound keyword relevance experiments from original eLife research abstracts. It also provides a transparent lexical diagnostic that penalizes unsupported and duplicated terms. The diagnostic is not a neural adapter, universal factual proof or a registered evaluator.

## Current state

The pinned complete inventory contains 19,699 v1 XML files. The same deterministic sample was expanded to 4,096 files without deleting or redownloading the previous 2,048. All selected blobs were verified. It yielded 1,783 licensed research-article-tagged XML records with original author keywords. Of these, 1,503 whole title/abstract/author-keyword renderings fit the 384-token budget. Nothing is cropped to keep an annotation.

Boundary-aware proposal construction finds 268 reciprocal contrasts. Thirty-one original-text reviews reject ambiguous directions. Thirty-two contrasts were accepted as AI-assisted comparative centrality/informativeness preferences, producing 64 source-derived pairs. Another 205 proposals are unreviewed. The unrejected proposal graph's capacity is not a count of proven valid contrasts. These are sample and recipe counts, not the annotation rate of the full eLife corpus.

No approved dataset or trained neural candidate exists. Current reviewed data has 20 train pairs over nine components, two validation pairs over one component and 42 audit pairs over fourteen components. Donor/source groups sharing normalized answers are merged before splitting. The assembler refuses the insufficient 64/16/64 counts rather than ignoring overlap or manufacturing independence. It does not issue an approved manifest.

Source-only public CI run 44624204582 passed 31 tests at source commit 23a587e28b16dd0024c8a0393c5acd11f5bcfcb1. Its actual source-result artifact is distinct from neural quality evidence. No trained runtime score, registration or qualification is claimed. New remote-contract tests are source-only synthetic controls, not a measured runtime report.

## Measurement boundary

Author keywords are original annotations, not exhaustive gold lists. Unannotated phrases can be useful. The proposal file is not training data. Opposite centrality preferences need explicit reasons from the exact retained abstracts. AI-assisted reviews remain labeled AI-assisted. They are not human reviews.

DOI, normalized author and duplicate abstract components are formed before token filtering. Topic membership is retained for the whole component. Approved reciprocal donors are joined before splitting. The prepared audit holds Neuroscience-containing components out. Formats are plain text, JSON arrays and JSON keyword objects with unchanged raw strings.

The node's renderer is exactly `### Task:\n<prompt>\n\n### Answer:\n<answer>`. The runnable identity checker follows the primary `BatchIdentity.Equal`: case-insensitive hex comparison without stripping prefixes. The primary executor emits an unprefixed adapter hash while the registration artifact uses a prefixed Keccak string. They are different wire representations.

## Verification and build path

`verify.sh` compiles Python source and runs source/parser/group/baseline/wire controls. `wire_probe.go` exercises the primary wire comparison without loading a model. `packagecheck.go` preserves the primary static tensor and finiteness checks with this family's fresh live class loaded only in its isolated process.

`assemble.py` accepts only explicitly reviewed preferences. It retains source grants, complete source annotations and review reasons. It rejects duplicate/conflicting review IDs and stale declared counts before assembly. It refuses empty or undersized splits and normalized source/answer overlap. The current minimum is 64 train pairs, 16 validation pairs and 64 audit pairs over at least 32/8/32 independent components. The frozen shortcut coefficients use integer arithmetic fitted only on training rows. Reproduction consumes their serialized coefficients, not a new floating-point fit.

`remote.py` refuses model operations outside the controller's public GitHub repository. It verifies every source claim against the full pinned inventory, including repeat claims for cached sources. It binds the exact prompt and response strings to the original construction. It rechecks minimum independent counts, runs the unmodified primary dataset gate and token conformance before training a fresh rank-8 LoRA. Validation selects the saved candidate. A seed-matched untrained initialization is exported separately. This training stage does not read audit metrics or publish an accuracy record.

The separate workflow uses standard public CPU train and audit jobs, hash-locked dependencies and read-only permissions. Each job has a 150-minute outer timeout. Its supervisor limits each process tree to 6,900 seconds and the lower of 10 GiB RSS or 75% of runner RAM. The same licence policy stamps successful exports and failed saved checkpoints. No provider, signer, SSH or deployment credential is needed by this module.

All supporting code and pinned model/dependency manifests live inside `verification/owners/keyword_extraction/`. No sibling training module or private checkout is required. The owned workflow is `evaluator-owner-keyword_extraction.yml`. Its source mode runs the packaged source-only gate on a standard public CPU runner without loading a neural model. Train/audit mode refuses missing or failed approved data.

## Remaining gates

The current reciprocal recipe has enough theoretical source capacity but needs many more clear reviews, not weaker tests. The owner must establish sufficient grouped preferences and pass source/shortcut gates before dispatching the pilot through the installed automatic action adapter. Publication and qualification use the adapter's independent byte/gate checks, not a human review wait.

`audit_remote.py` implements one audit of the frozen candidate in the real exported runtime across all three formats. It compares seed-matched untrained and strongest frozen baselines, repeats exact score bits in fresh processes and obtains input roots from the primary Go implementation. Four wrong-identity requests must return the executor's real identity, not echoes. Negative tokens must produce an actual engine error. It rejects changed package, data, config, foundation-layout or commit identities. Grouped counts and a bounded-component uncertainty interval are reported with their sampling assumption. Required quality is at least 80% overall, 70% per observed label kind and length stratum, five points above initialization and ten above the strongest shortcut. The 65% data-shortcut ceiling is unchanged. These remote gates have not run.

The ready-report emitter combines actual parser, audit, runtime and source evidence only after these gates pass. Its per-label metrics describe observed source label_kind values, not absolute relevance labels. It binds eight actual fixed array-format vectors, asset hashes and the audit/source commit. No registration-ready.json has been produced by a real job.

Full deployed daemon string handling is still unverified. The live child currently has null request and response schemas. This does not automatically block a judgment adapter, but it does not establish a complete app request path. Live endpoint validation and fresh independent node qualification still require actual passing artifacts and adapter checks. Identical serving mode remains integration work before registration. A source-held-out score is not a canonical win.

## Terms

Our additions use LicenseRef-zkasuran-SAND-1.0. Original eLife text and annotations keep their per-article CC BY 3.0, CC BY 4.0 or CC0 terms. The smoke fixture retains its original XML permissions. Full notices and source terms are in `licences.json` and the granting clauses are recorded in `DATA-SOURCES.md`. ModernBERT retains Apache-2.0 and the Telegraph reference kit retains MIT. No endorsement is implied.
