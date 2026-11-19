# Source-bound obligation and exception entailment

This pipeline trains and audits a clause-grounding reward adapter on bounded ContractNLI evidence. It checks identification, third-party sharing, copying permissions and retention exceptions. It continues a prior literal-citation diagnostic without reusing its exposed audits. Neural training and inference run only in this repository's bounded public CPU workflow.

The new source is ContractNLI, Yuta Koreeda and Christopher Manning (2021), CC BY 4.0. All 607 originals and 10,319 annotations are inventoried. Source labels are document-scoped. Complete annotated evidence spans are retained and bounded to 256 UTF-8 bytes. NotMentioned labels and incomplete list/conjunction fragments are excluded rather than relabelled as passage-level absence.

Generation two has 64 train pairs in 57 source groups, 16 validation pairs in 13 groups and 64 reserved audit pairs in 56 groups. Each hypothesis has both labels with equal counts within every split. Whole source hosts, SEC issuer CIKs and normalized duplicate originals form connected groups. Cross-split groups and normalized evidence overlaps are removed. Forty-three retired audit groups are excluded from every new split. All 165 previously exposed groups are excluded from the new audit. Source-derived data retains CC BY 4.0. Response templates are AI-assisted constructions, not freshly human-reviewed preferences.

## Source and quality gates

The first construction failed because a train-fitted hypothesis prior scored 100% on validation. Its inspected audit remains retired. Generation two passes the unchanged source gate: length, unigram overlap, answer-only bag-of-words, exact-quote and hypothesis-prior shortcuts all score 50% on training and validation. New audit metrics have not been inspected during development.

Training uses the pinned ModernBERT base, a seed-matched untrained LoRA/head and validation-only best-epoch selection. The independent audit runs the frozen trained and untrained packages through the actual reference foundation engine. Every claimed JSON/prose format requires 80% overall, 70% per source label, five points over untrained and ten points over the strongest frozen shortcut. A bare Unclear answer has no contrast and is not claimed as a validated response format.

The full current Go package parser checks all tensors, metadata, offsets and finite values against a fresh public class snapshot. The engine produces actual scores for every audit pair, repeats eight fixed vectors and exercises three literal HTTP identity refusals plus invalid token controls. A registration-ready report is emitted only with its actual pass/fail status. The node's fresh canonical decision remains separate.

## Run source contracts

```sh
bash verification/owners/contract_obligation_audit/verify.sh
```

The source gate uses the Python standard library. It reconstructs the complete hash-pinned source, verifies exact split fingerprints and frozen train-fitted integer baselines, then validates the owned JSON-encoded YAML workflow. Local tests cover literal rendering, strict JSON, finite score bits, bare hash echoes, actual HTTP failure controls, frozen baseline binding and quality thresholds. Authored local fixtures are not source training data or measured reference-engine vectors.

The source gate can fetch its own bounded public archive in a standalone checkout. An optional OWNER_SOURCE_ARCHIVE cache is hash-verified before use. It never loads a local model. The remote workflow has explicit train/audit/serve modes, ubuntu-24.04 runners, read-only repository access and bounded 240/115/330-minute job limits. Result artifacts bind actual run, commit, data, package, runtime and executable identities.

Successful and failed checkpoints retain the same SAND and upstream grant boundary. The licensed-checkpoint artifact includes all original notices. Serving accepts only exact immutable package/report URLs and uses the identical public executor. Only the public restricted lease is uploaded. The runner's forwarding key is ephemeral and no signer/provider key is supplied.

## Limits

Clauses and obligations are off-chain in the current live schema. A verdict-only answer carries no evidence for this narrow claim. All candidate responses say Unclear because no actual performance is established. ContractNLI cannot prove performance compliance, amend an agreement or resolve every entity alias. Source-host/issuer grouping is conservative but not a full beneficial-owner identity graph. Public reference-wire contracts are checked here; actual engine execution is still remote work.

Source checks do not establish model quality or a canonical result. Publication, remote model results and qualification have their own proof-bound actions. No successful model metric is filled from an expected value. See SECURITY.md and DATA-SOURCES.md for the source and transport limits.
