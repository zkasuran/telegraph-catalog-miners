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
