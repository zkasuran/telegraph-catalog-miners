# Find scholarly titles related to a research paper

SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0

This source pipeline prepares paper-disjoint scholarly title relevance
contrasts for ACADEMIC_SEARCH. Original RELISH expert annotations supply
article-relevance labels. OpenAlex supplies CC0 bibliographic metadata. Queries
and answers keep their original text and identifiers.

## What works

- Bounded source capture with original grant, checksum and receipt checks.
- Complete annotation inventory, disagreement exclusion and exact metadata joins.
- Entity/title/DOI alias grouping and full-token visibility screens.
- Explicit AI-assisted source-review boundary before any final split.
- Frozen 128/32/64 original groups with integer train-fitted controls.
- Literal node wire identity checks and a remote-only ModernBERT scorer stage.
- Compiled exact-family full-parser integration with actual upstream failure controls.
- A bounded public CPU workflow with independent audit, licensed export and actual reference-engine checks.
- Verified-package serving mode with the existing restricted public-lease contract.

## Run the source gate

```sh
bash verify.sh
```

The gate compiles source syntax, validates the model contract and runs parser,
failure and property controls without importing neural libraries. It also tests
a small public source-derived smoke fixture. Artificial test fixtures never
become training or audit data.

## Data and model boundary

RELISH human annotations compare complete papers. Our generated pairs are not
human pairwise preference votes. Title-only alignment was screened with explicit
AI-assisted review before the fixed source split. The only supported renderings are
the complete title, compact JSON and prose, each retaining the same title/DOI.
Total result counts are omitted because the source does not establish them.

Model imports sit behind the standard public CPU GitHub workflow guard. No model
loading, fitting or inference is part of local verification. The scorer stage
selects by validation loss. A separate audit process measures that checkpoint
once, exports the licensed delta and scores all three formats with the actual
pinned reference engine. Only complete passing measurements can produce
registration-ready.json. The pipeline itself never signs or registers.

## What is not finished

The frozen source is not a qualified model. Remote training, independent audit,
trained-package checking, exact reference-engine scores and serving have not run.
Their implementation is prepared and source-tested, not reported as a model pass.
No new canonical registration or measured neural quality is claimed.

The owned workflow uses ubuntu-24.04 with one CPU job and an explicit 350-minute
ceiling. Its train mode preserves licensed results even after a failed gate.
Its serve mode checks immutable package/report bytes and real public vectors
before generating a restricted five-hour public forwarding lease. Only that
public lease is uploaded. No operator key or private forwarding key is uploaded.

## Licence and provenance

Our additions use LicenseRef-zkasuran-SAND-1.0. Original CC0 annotations and
metadata retain their grants. Reference kit MIT and base-model Apache terms are
retained in full. Read NOTICE, DATA-SOURCES.md and SECURITY.md before using data.
# Development update after the first measured loss

The first frozen candidate passed the full parser and reference runtime but failed quality. Its exposed audit is retired and remains preserved as historical evidence.

Development mode fits only the original training split. It warms the reward head with train-only length-stratum weighting then trains rank-16 LoRA with one original-group format view per epoch, rotated across title, JSON and prose. Mean validation loss across all formats selects the frozen checkpoint. Development output is not registration-ready and is not an independent audit.

A later qualification needs a clean source/entity-disjoint audit. The original grants, baseline coefficients and quality gates remain. Model work runs only on the bounded public CPU workflow, never in the local source gate.
