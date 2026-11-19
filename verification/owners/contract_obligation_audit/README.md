# Source-bound obligation and exception entailment

This pipeline trains and audits a clause-grounding reward adapter on bounded ContractNLI evidence. It checks identification, third-party sharing, copying permissions and retention exceptions. It continues a prior literal-citation diagnostic without reusing its exposed audits. Neural training and inference run only in this repository's bounded public CPU workflow.

The new source is ContractNLI, Yuta Koreeda and Christopher Manning (2021), CC BY 4.0. All 607 originals and 10,319 annotations are inventoried. Source labels are document-scoped. Complete annotated evidence spans are retained and bounded to 256 UTF-8 bytes. NotMentioned labels and incomplete list/conjunction fragments are excluded rather than relabelled as passage-level absence.

Generation four has 128 train pairs in 82 source groups, 30 validation pairs in 25 groups and 64 reserved audit pairs in 61 groups. Each hypothesis has both labels with equal counts within every split. Whole source hosts, SEC issuer CIKs and normalized duplicate originals form connected groups. Cross-split groups and normalized evidence overlaps are removed. Ninety-nine retired audit groups are excluded from every new split. The generation-three audit never executed, so its original source IDs and groups remain fixed, unconsumed and excluded from every fit/selection step. Source-derived data retains CC BY 4.0. Response templates are AI-assisted constructions, not freshly human-reviewed preferences.

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

Generation three's all-format validation correctly refused its weak candidate before audit. Generation four resumes that retained source-trained checkpoint, broadens source diversity and uses direct statements versus their logical complements, not only supports/refutes tags. It trains both formats for at most 16 epochs with the same planned 512 optimizer steps as the earlier 32-epoch smaller-source experiment. Selection protects worst-format validation accuracy and every quality gate remains unchanged. A changed tensor payload is required. No new passing model is claimed until actual remote results exist.

## Raw scoring length and stronger comparisons

The scorer consumes raw judgment strings through the published 512-token contract. Complete annotated source contexts are retained up to 768 UTF-8 bytes. An authoritative tokenizer preflight found 980 eligible original records, with at most 430 rendered tokens for direct/complement claims and no neural modules loaded. Remote training and audit recheck every actual token vector and refuse any truncated claim. This does not claim that the separate typed miner slot transports a contract longer than 256 bytes or that links are dereferenced.

Source integrity shortcuts remain at 50% in training/development. Frozen source-aware negation and exception baselines reach 64.0625% and 67.96875% on training, 66.6667% and 63.3333% on validation. These are semantic comparison baselines, not answer-only leakage. They strengthen the unchanged model requirement of ten points over the strongest baseline and do not replace any source-integrity gate. No audit model or baseline score has been read for candidate selection.
