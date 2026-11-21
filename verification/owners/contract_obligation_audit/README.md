# Source-bound obligation and exception entailment

This pipeline trains and audits a clause-grounding reward adapter on bounded ContractNLI evidence. It checks identification, third-party sharing, copying permissions and retention exceptions. It continues a prior literal-citation diagnostic without reusing its exposed audits. Neural training and inference run only in this repository's bounded public CPU workflow.

The source is ContractNLI, Yuta Koreeda and Christopher Manning (2021), CC BY 4.0. All 607 originals and 10,319 annotations are inventoried. Source labels are document-scoped. Complete annotated evidence spans are retained within the documented raw 512-token scoring surface. Quote-format contexts are bounded to 768 UTF-8 bytes. Explicit training-only citation rows allow up to 1600 bytes without changing development or audit evidence bounds. NotMentioned labels and incomplete list/conjunction fragments are excluded rather than relabelled as passage-level absence.

Generation eight has 396 train pairs in 212 source groups, 30 validation pairs in 25 groups and 64 reserved audit pairs in 61 groups. The original 356 training records are unchanged; 40 wider complete contexts extend training coverage only. Every audited hypothesis has both source labels. Whole source hosts, SEC issuer CIKs and normalized duplicate originals form connected groups. Cross-split groups and normalized evidence overlaps are removed. Ninety-nine retired audit groups are excluded from every new split. The reserved audit remains unconsumed and excluded from fitting and selection. Source-derived data retains CC BY 4.0. Response templates are AI-assisted constructions, not freshly human-reviewed preferences.

## Source and quality gates

The first construction failed because a train-fitted hypothesis prior scored 100% on validation. Its inspected audit remains retired. Before its independent neural audit, generation two passed the unchanged source gate: length, unigram overlap, answer-only bag-of-words, exact-quote and hypothesis-prior shortcuts all scored 50% on training and validation. Audit metrics were not inspected during candidate selection. That generation's audit is now retired after the measured quality failure described below.

Training uses the pinned ModernBERT base, a seed-matched untrained LoRA/head and validation-only best-epoch selection. The independent audit runs the frozen trained and untrained packages through the actual reference foundation engine. Every claimed JSON/prose format requires 80% overall, 70% per source label, five points over untrained and ten points over the strongest frozen shortcut. A bare Unclear answer has no contrast and is not claimed as a validated response format.

The full current Go package parser checks all tensors, metadata, offsets and finite values against a fresh public class snapshot. The engine produces actual scores for every audit pair, repeats eight fixed vectors and exercises three literal HTTP identity refusals plus invalid token controls. A registration-ready report is emitted only with its actual pass/fail status. The node's fresh canonical decision remains separate.

## Run source contracts

```sh
bash verification/owners/contract_obligation_audit/verify.sh
```

The source gate uses the Python standard library. It reconstructs the complete hash-pinned source, verifies exact split fingerprints and frozen train-fitted integer baselines, then validates the owned JSON-encoded YAML workflow. Local tests cover literal rendering, strict JSON, finite score bits, bare hash echoes, actual HTTP failure controls, frozen baseline binding and quality thresholds. Authored local fixtures are not source training data or measured reference-engine vectors.

The source gate can fetch its own bounded public archive in a standalone checkout. An optional OWNER_SOURCE_ARCHIVE cache is hash-verified before use. It never loads a local model. The remote workflow has explicit train/audit/serve modes, ubuntu-24.04 runners, read-only repository access and bounded 340/115/330-minute job limits. Result artifacts bind actual run, commit, data, package, runtime and executable identities.

Successful and failed checkpoints retain the same SAND and upstream grant boundary. The licensed-checkpoint artifact includes all original notices. Serving accepts only exact immutable package/report URLs and uses the identical public executor. Only the public restricted lease is uploaded. The runner's forwarding key is ephemeral and no signer/provider key is supplied.

## Limits

Clauses and obligations are off-chain in the current live schema. A verdict-only answer carries no evidence for this narrow claim. All candidate responses say Unclear because no actual performance is established. ContractNLI cannot prove performance compliance, amend an agreement or resolve every entity alias. Source-host/issuer grouping is conservative but not a full beneficial-owner identity graph. Public reference-wire contracts are checked here; actual engine execution is still remote work.

Source checks do not establish model quality or a canonical result. Publication, remote model results and qualification have their own proof-bound actions. No successful model metric is filled from an expected value. See SECURITY.md and DATA-SOURCES.md for the source and transport limits.

## First measured model result and current recovery

Run 44564516958 completed full parsing and actual reference-engine execution, but failed quality. JSON pair accuracy was 39.0625%, prose 46.875% and seed-matched untrained 46.875% in both views. Native validation had already failed at 50%. Those audit rows are permanently retired and the old source generation is now archival, not a new qualifying dataset.

Later all-format validation refused weak candidates before audit. The latest measured rank-growth run retained all 88 factor pairs and learned 1081344 new output coefficients, but native validation still failed at 73.33% JSON and 70% prose. The current source trains both formats for at most six epochs with retained rank-sixteen capacity. Selection protects worst-format validation accuracy and every quality gate remains unchanged. A changed tensor payload is required. No new passing model is claimed until actual remote results exist.

## Raw scoring length and stronger comparisons

The scorer consumes raw judgment strings through the published 512-token contract. Complete annotated source contexts are retained up to 768 UTF-8 bytes. An authoritative tokenizer preflight found 980 eligible original records, with at most 430 rendered tokens for direct/complement claims and no neural modules loaded. Remote training and audit recheck every actual token vector and refuse any truncated claim. This does not claim that the separate typed miner slot transports a contract longer than 256 bytes or that links are dereferenced.

Source integrity shortcuts remain at 50% in training/development. The citation-training subgroup is gated separately at the same cutoff. Frozen source-aware negation and exception baselines reach 60.3535% and 62.6263% on training, 66.6667% and 63.3333% on validation. These are semantic comparison baselines, not answer-only leakage. They strengthen the unchanged model requirement of ten points over the strongest baseline and do not replace any source-integrity gate. No audit model or baseline score has been read for candidate selection.

## Diagnose development failures without exposing the audit

The source-only diagnostic verifies literal prompt/answer reconstruction, grouped overlap, original label coverage and optional token bounds. It refuses audit rows even if a file is renamed as validation. It never loads a model or selects a checkpoint.

```sh
python3 verification/owners/contract_obligation_audit/development_diagnostics.py \
  --train contract-run/data/train.jsonl \
  --validation contract-run/data/validation.jsonl \
  --out development-structure.json
```

These paths are outputs of the source pipeline in a standalone checkout. Optional `--tokenizer` accepts only the pinned tokenizer JSON and performs tokenization without padding or truncation. Optional `--wire` with `--package-keccak` diagnoses ties and cross-format sign disagreements in supplied literal batch responses. It checks bare hash echoes and finite score bits, but supplied responses are not a newly verified reference-engine run. No diagnostic produces registration-ready status. The source release gate exercises primary and failure controls for these boundaries.

The public CPU training workflow also has a validation-only reference capture step after a training freeze exists. `validation_capture_remote.py` verifies the immutable run, commit, source and two package identities, runs the full parser and scores validation originals through the pinned reference executor. It records actual returned bare-hash echoes and score bits for JSON/prose, ties and per-label/per-hypothesis counts. No audit rows are loaded and no model is selected by this step. Its report is `VALIDATION_RUNTIME_OBSERVED`, never `REGISTRATION_READY`.

Capture has a 20-minute internal deadline and a 25-minute workflow step limit. Request timeouts are capped by the remaining capture budget. Native validation gates and the independent audit remain unchanged. A failed native validation still leaves its frozen candidate available for this development diagnosis; it does not make the audit job eligible. Actual capture/model results require a bounded public workflow run and are not claimed by local fixture tests.

The client deadline includes scorer-lock wait and the full response duration, not only socket inactivity. A bounded watchdog interrupts a slow-drip response even with `Connection: close`. Connection-construction failures release the scorer lock. Validation capture also caps reference startup by its remaining budget. Authored loopback HTTP fixtures reproduce these failure paths without loading a model; no neural performance claim comes from them.
