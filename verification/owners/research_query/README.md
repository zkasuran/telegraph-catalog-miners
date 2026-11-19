# Check scientific claims against the excerpt

This RESEARCH_QUERY experiment scores whether a short scientific claim agrees
with a complete supplied excerpt. It does not certify scientific truth or
retrieve papers. Source text comes from independently pinned eLife originals.
Questions and preferences are AI-assisted constructed work, not human-reviewed.

## Verified source build

| Split | Article components | Mirrored preference pairs |
|---|---:|---:|
| Training | 64 | 128 |
| Validation | 16 | 32 |
| Sealed audit | 32 | 64 |

The complete article index contains 19,699 v1 files. A fixed SHA-256-ordered
sample read 6,144 XML files and found 118 compact candidate articles. It is
not a full-XML population audit. DOI, article page, normalized title, evidence
and both claim variants are connected before splitting. All complete rendered
inputs fit in at most 154 tokens. Request strings are at most 256 UTF-8 bytes.

The unchanged primary dataset gate passed on 128 training pairs: 100% unique
negatives, 50% length ordering, zero negative leakage and 45.26% answer-only
surface-probe accuracy. Seven immutable validation shortcut controls score 50%.
The audit has not been used for model or shortcut selection.

## What is real and what is pending

| Item | State |
|---|---|
| Original XML, per-article grant and complete excerpt | Source-bound and retained |
| Literal source comparator and CLI | Useful baseline for the exact restricted grammar |
| Remote CPU training and export code | Prepared, not remotely executed |
| FP32 package and repeated runtime bits | Not produced or verified yet |
| Live counterparty judgment-body selection | Controller must confirm with a fresh literal wire probe |
| Qualification or canonical status | Not attempted |

The comparator is exact for a restricted supplied-premise grammar. A neural
model cannot be claimed to improve over that comparator without evidence.
Its source-held-out score must not be called broad research quality.

## Source quickstart

Run from the catalog checkout after the controller applies the reviewed patch:

```sh
python3 -W error verification/owners/research_query/verify.py
```

This requires PyYAML. Remote training dependencies pin PyYAML 6.0.3. It parses
the workflow, validates source identities, runs adversarial tests and exercises
the actual literal CLI. It downloads one pinned original if the smoke cache is
missing. It never loads a neural model. `remote.py` refuses ordinary local use.

## Remote candidate procedure

The automatic action adapter owns the proof-bound dispatch of the `evaluator-owner-research_query.yml`
workflow. It uses the standard Ubuntu 24.04 CPU runner, four CPU threads and a
330-minute outer bound. Hash-locked dependencies and the unchanged data gate
run before training. The job fits three epochs on training only and chooses
the lowest validation loss. It restores that checkpoint before one fixed audit.

The audit measures 64 correlated queries from 32 independent original groups.
The job reports both direction subsets, untrained initialization, frozen
shortcut controls, removal of decisive evidence and the meaningful symbolic
task baseline. It exports a fresh rank-8 LoRA package, runs the full primary
Stage 1 checker and compares native scores with the primary FP32 executor.
Repeated inputs, a fresh process and wrong foundation/tokenizer/package
identities are required controls. A report binds commit, workflow run, data,
package, foundation, tokenizer, engine and the contacted node identity.

Publication, dispatch and registration use only the installed automatic action adapter. No credentials enter this pipeline. A passing source report still needs fresh live endpoint checks and independent canonical node qualification.

## Licence

New code uses LicenseRef-zkasuran-SAND-1.0. Original eLife text keeps each
article's CC BY 3.0, CC BY 4.0 or CC0 grant. Full source attribution and change
notices are retained in every row and in successful or failed checkpoints.
MIT kit code and Apache-2.0 ModernBERT terms remain attached. See NOTICE and
DATA-SOURCES.md. SAND does not restrict the original source grants.

## Standalone owner pipeline

All dependencies on the paid-for public harness are pinned under support/. The source gate runs without Torch or model loading. The remote job trains on fixed grouped originals, selects on validation only and audits both literal prose and raw JSON answer bodies. REGISTRATION_READY is emitted only after actual parser, reference-engine, per-label and baseline gates pass. Serving mode uses the same source commit and exact immutable delta, then publishes only a restricted public forwarding lease.
