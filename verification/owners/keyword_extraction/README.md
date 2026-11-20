# Keyword centrality pilot

This module prepares source-bound keyword relevance experiments from original eLife research abstracts. It also provides a transparent lexical diagnostic that penalizes unsupported and duplicated terms. The diagnostic is not a neural adapter, universal factual proof or a registered evaluator.

## Current state

The pinned complete inventory contains 19,699 v1 XML files. The same deterministic sample was expanded to 4,096 files without deleting or redownloading the previous 2,048. All selected blobs were verified. It yielded 1,783 licensed research-article-tagged XML records with original author keywords. Of these, 1,503 whole title/abstract/author-keyword renderings fit the 384-token budget. Nothing is cropped to keep an annotation.

The original boundary-aware recipe finds 268 reciprocal proposals. An expanded search examines all 1,503 intact records and 1,128,753 source pairs using 2,932 original annotated literal phrases. It preserves the original proposal IDs and adds 641 unreviewed proposals, for 909 total. Each new phrase has an original annotation witness from one of the two articles and is literal in both complete abstracts. No annotation membership implies a preference.

The source review snapshots preserve all earlier decisions. The approved pilot contains 82 accepted AI-assisted comparisons (164 pairs). Sixty ambiguous or synonymous inversions were rejected. Four reciprocal cross-topic precision comparisons are confined to training and retain both original own/donor annotations and source grants. They provide eight of 68 training pairs, below the unchanged 25% donor ceiling. Earlier 64-pair and 114-pair snapshots remain intact. These are bounded-sample counts, not the annotation rate of the full eLife corpus.

The source data gate now passes with 68 train pairs over 32 components, 16 validation pairs over eight components and 80 audit pairs over 33 components. Donor/source groups sharing normalized answers are merged before splitting. Validation and audit source bytes were frozen before adding the final training material and remain unchanged. Original XML, grants, annotation witnesses, normalized overlap and all-format token-fit checks pass. No neural training or model quality is claimed from this source gate.

Source-only public CI run 44662981154 passed 34 tests at source commit 72f45a429eaf4a03beb26f6f86d456e9202e11ec. Its actual source-result artifact is distinct from neural quality evidence. No trained runtime score, registration or qualification is claimed. New expansion tests are source-only synthetic controls, not source or model preferences.

## Measurement boundary

Author keywords are original annotations, not exhaustive gold lists. Unannotated phrases can be useful. The proposal file is not training data. Opposite centrality preferences need explicit reasons from the exact retained abstracts. AI-assisted reviews remain labeled AI-assisted. They are not human reviews.

DOI, normalized author and duplicate abstract components are formed before token filtering. Topic membership is retained for the whole component. Approved reciprocal donors are joined before splitting. The prepared audit holds Neuroscience-containing components out. Formats are plain text, JSON arrays and JSON keyword objects with unchanged raw strings.

The node's renderer is exactly `### Task:\n<prompt>\n\n### Answer:\n<answer>`. The runnable identity checker follows the primary `BatchIdentity.Equal`: case-insensitive hex comparison without stripping prefixes. The primary executor emits an unprefixed adapter hash while the registration artifact uses a prefixed Keccak string. They are different wire representations.

## Verification and build path

`verify.sh` compiles Python source and runs source/parser/group/baseline/wire controls. `wire_probe.go` exercises the primary wire comparison without loading a model. `packagecheck.go` preserves the primary static tensor and finiteness checks with this family's fresh live class loaded only in its isolated process.

`assemble.py` accepts only explicitly reviewed preferences. It retains source grants, complete source annotations and review reasons. It rejects duplicate/conflicting review IDs and stale declared counts before assembly. It refuses empty or undersized splits and normalized source/answer overlap. The current minimum is 64 train pairs, 16 validation pairs and 64 audit pairs over at least 32/8/32 independent components. The frozen shortcut coefficients use integer arithmetic fitted only on training rows. Reproduction consumes their serialized coefficients, not a new floating-point fit.

`heldout_freeze.py` binds exact validation/audit bytes and counts before model selection. Approved export and remote loading require that same freeze. Capped precision cases use fully reviewed cross-topic contexts, not a claim that word absence universally proves irrelevance. Their donor source identity and attribution are verified alongside the focal source, including cross-split identity checks. The primary neural dataset gate is still run unchanged on the public CPU runner before any model preparation.

`expand_proposals.py` searches each complete retained source record with a bounded phrase trie. It retains nested matches, refuses morphological substrings and checks every new phrase against the primary literal rule. It writes a separate snapshot and retains prior reviews supplied through `--reviews`. `proposal_matching.py` orders unreviewed comparisons by disjoint source and answer resources. Its optional split priority stratifies review work under the existing fixed split rule, not model outcomes. Packing is not approval or a maximum matching. `review_view.py --packed` refuses stale source/review hashes before displaying the exact full abstracts.

`remote.py` refuses model operations outside the controller's public GitHub repository. It verifies every source claim against the full pinned inventory, including repeat claims for cached sources. It binds the exact prompt and response strings to the original construction. It rechecks minimum independent counts, runs the unmodified primary dataset gate and token conformance before training a fresh rank-8 LoRA. Validation selects the saved candidate. A seed-matched untrained initialization is exported separately. This training stage does not read audit metrics or publish an accuracy record.

The separate workflow uses standard public CPU train and audit jobs, hash-locked dependencies and read-only permissions. Each job has a 150-minute outer timeout. Its supervisor limits training/audit trees to 6,900 seconds and the lower of 10 GiB RSS or 75% of runner RAM. The same licence policy stamps successful exports and failed saved checkpoints. No provider or signer credential is needed by this module.

Serving mode has a 350-minute outer ceiling and an exact-entrypoint supervisor capped at 19,800 seconds, covering preparation plus a five-hour restricted lease. `serve.py` requires passing immutable report/package bytes, actual successful audit metadata, the identical executor binary and matching reference score bits before generating an ephemeral forwarding key on the public runner. It uses the existing `lease_for` contract with candidate/runtime/source proof hashes bound to the same combined report. Only the public restricted lease is uploaded. No private key or source/audit corpus is uploaded as a serving artifact. Lease renewal and fresh canonical observation remain owner/controller operations, not new model registrations.

All supporting code and pinned model/dependency manifests live inside `verification/owners/keyword_extraction/`. No sibling training module or private checkout is required. The owned workflow is `evaluator-owner-keyword_extraction.yml`. Its source mode runs the packaged source-only gate on a standard public CPU runner without loading a neural model. Train/audit mode refuses missing or failed approved data.

## Remaining gates

The source count, grouped-overlap, grant and immutable baseline checks now pass. Neural training, full package parsing and independent reference-engine quality measurement still need a bounded public CPU job through the installed automatic action adapter. Publication and qualification use the adapter's independent byte/gate checks, not a human review wait.

`audit_remote.py` implements one audit of the frozen candidate in the real exported runtime across all three formats. It compares seed-matched untrained and strongest frozen baselines, repeats exact score bits in fresh processes and obtains input roots from the primary Go implementation. Four wrong-identity requests must return the executor's real identity, not echoes. Negative tokens must produce an actual engine error. It rejects changed package, data, config, foundation-layout or commit identities. Grouped counts and a bounded-component uncertainty interval are reported with their sampling assumption. Required quality is at least 80% overall, 70% per observed label kind and length stratum, five points above initialization and ten above the strongest shortcut. The 65% data-shortcut ceiling is unchanged. These remote gates have not run.

The ready-report emitter combines actual parser, audit, runtime and source evidence only after these gates pass. Its per-label metrics describe observed source label_kind values, not absolute relevance labels. It binds eight actual fixed array-format vectors, asset hashes and the audit/source commit. No registration-ready.json has been produced by a real job.

Full deployed daemon string handling is still unverified. The live child currently has null request and response schemas. This does not automatically block a judgment adapter, but it does not establish a complete app request path. Live endpoint validation and fresh independent node qualification still require actual passing artifacts and adapter checks. The serving path is source-tested but has not run a real model or endpoint. A source-held-out score is not a canonical win.

## Terms

Our additions use LicenseRef-zkasuran-SAND-1.0. Original eLife text and annotations keep their per-article CC BY 3.0, CC BY 4.0 or CC0 terms. The smoke fixture retains its original XML permissions. Full notices and source terms are in `licences.json` and the granting clauses are recorded in `DATA-SOURCES.md`. ModernBERT retains Apache-2.0 and the Telegraph reference kit retains MIT. No endorsement is implied.
