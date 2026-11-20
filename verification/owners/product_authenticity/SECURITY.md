# Threat model

| Attack | Defence | Test |
|---|---|---|
| Seller copies an archive ID | Explicit archived scope and exact product echo | copied_identifier_new_listing_and_wrong_echo_refused |
| Missing row or cheap price is called counterfeit | Unknown remains Inconclusive | preserved low-price, absent-row and copied-barcode tests |
| Source says falsified but spectrum resembles standard | Keep manufacturer-assessed source label | preserved_source_false_negative_not_relabelled |
| Withheld spectra are filled in | No spectrum field or synthetic standard ID | no_withheld_spectra_or_extra_standard_ids |
| Source grant or range changes | Hash seal plus granting/range checks | source_grant_and_identifier_range_mutations_fail |
| Duplicate JSON keys or deep/oversized replies | Strict JSON, byte limits and bounded envelope | preserved strict JSON and random-byte tests |
| Repeated measures or papers pretend independence | Two conservative source components and linked cohorts | same_source_entity_cohort_not_three_new_studies |

The source-only release gate has no neural imports, runtime inference or network
access. Bundled Pyflakes is static lint. Archived source labels are not legal
chain-of-custody certificates or fresh physical-product proof. Newton S18 remains
Inconclusive because chemistry was obtained from another package. Unknown
Inconclusive's false boolean means no positive authentic judgment, not counterfeit.

The actual deployed caller and raw/shaped response mapping are unverified. This
diagnostic claims exact JSON archive coverage only. Its lookup and G/F identifier
shortcuts prevent treating successful control scores as semantic model quality.
No registration-ready artifact is emitted. Source reports explicitly state that
model qualification, independent audit and reference-engine execution are absent.

## Development model boundary

Neural entrypoints require the exact owned workflow_dispatch standard public GitHub runner. Local source gates refuse neural imports. Byte-bounded source snapshots, fixed source hashes, grouped source/entity/time separation and exact raw renderer bytes are checked before training. Remote source gates run unchanged shortcut controls. Model selection uses validation loss only; no audit exists. Runtime reports come from the pinned full parser and reference engine, with literal bare adapter-hash echoes, repeated bits and three negative identity controls. No registration-ready file is emitted by development mode. All saved successful or failed checkpoint containers carry original upstream and SAND notices without changing tensor payloads.

## Independent audit admission

The additive source gate tests complete specimen inventory, original QMN identity, split-batch suffixes, column-bound batch extraction, attribution, compatible grants and refusal of quality/lookup-to-authenticity promotion. Its full-workbook inventory prevents treating an analytical-method attachment as a specimen corpus. See audit_admission/SECURITY.md. No model scoring occurs in this gate and no independent groups are fabricated from sample counts.

## Broader response and framing controls

The new source gate rejects normalized/source/entity/time/formulation overlap, incorrect literal request framing, oversize product bytes and unknown-to-counterfeit promotion. Remote mode alone may train or run the model. It retains the primary dataset gate and immutable train-fitted baselines. Validation loss may choose one candidate; no audit exists. A 256-byte product and its 340-byte typed question were executed against the actual primary decoder/request helper without a model. This is library evidence, not a deployed caller trace.

## Frozen-token and selected-checkpoint recovery

The package checker now binds the expected token ceiling explicitly (256 legacy or 512 broad), while retaining full primary Stage 1, byte identity, rank and no-model-loaded checks. Tiny Go controls reject the wrong expected ceiling, changed hash, missing Stage 1, model loading and rank changes. Recovery refuses changed data/package hashes and validation-fitted baselines. Its inverse of the pinned exporter requires byte equality for every selected tensor. No local model operation or new candidate selection is allowed.
