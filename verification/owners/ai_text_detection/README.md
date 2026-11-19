# Scientific text-origin reward pilot

This pipeline learns to rank an origin verdict for a short scientific passage.
It compares named-author journal controls with disclosed GPT-2 continuations.
Its independent audit uses DistilGPT2, a related held-out checkpoint, on separate
articles and a later publication-time block. It is not a general authorship oracle.

## Run the source-only gate

Run `bash verification/owners/ai_text_detection/verify.sh` from a public checkout.
The gate uses Python's standard library. It checks the complete source manifest,
individual grants, source groups, literal answer formats, finite score bits and
failure controls. It never loads a model or downloads weights locally.

## What is real

The pinned inventory contains 32,141 versioned XML files and 19,701 article IDs.
A fixed bounded scan of 960 candidates selected 224 article groups: 128 training,
32 validation and 64 audit. Their individual CC BY clauses and citations remain
in `sources.json`. These are sample counts, not corpus-wide quality measurements.

The answers and preference pairs are constructed by AI-assisted code, not human
preference votes. Human controls are pre-2019 journal text attributed to named
authors. The production history of each passage was not independently certified.
Synthetic origin is recorded from an actual remote generation, never inferred
from a model's output label.

## Remote pipeline

Only the controller-owned public CPU workflow runs the generators and reward
model. Development generation precedes clean rank-8 LoRA training. Validation
selects the saved epoch. A byte-hashed candidate and integer train-fitted baseline
counts freeze before audit generation. No audit score chooses the candidate.

The audit scores all JSON, prose and bare-label responses in the pinned Telegraph
reference engine. It also measures a seed-matched untrained package in that engine.
Passing requires 80% overall, 70% per origin, five points over untrained and ten
points over the strongest length, overlap or answer-only shortcut. Strict token
identity, three wrong identity hashes, a prefixed-hash control, invalid-token
refusal and fresh-process identical score bits are required.

Only real passes emit `registration-ready.json`. Publication, endpoint activation
and testnet registration use the automatic action adapter. No signer is present
in this pipeline. The serving runner uploads only a restricted public lease.

## Continued model development

The initial two-epoch model passed its source and full package parser gates but
scored only 54.69% on validation. Its attempted audit generation failed before
any reference-engine quality score. That package is retained, not registered.

Revision 2 continues its exact licensed adapter with at most eight epochs,
accumulation 8 and learning rate 0.0003. The backbone, rank, source fit groups and
validation groups remain fixed. The original optimizer was not exported, so a
new bounded AdamW optimizer is explicit. A fresh 64-group audit replaces the
quarantined original block. Every supported validation format must pass before
that audit can be generated or measured. More updates are a testable improvement
path, not a claim that the next model passes.

The first continuation attempt stopped at the unchanged repeated-negative data
gate before loading a checkpoint. Its bare labels leaked into the internal
holdout. Each bare opposite now occurs once and the actual source-generated
development bytes are restored with their producer and split roles unchanged.
The independent audit remains sealed. This fixes construction, not a model score.

The repaired continuation passed the real data gate and reached 82.81% JSON and
89.06% prose validation. Bare labels reached 71.88%, so it was not registered and
the audit stayed sealed. The next continuation adds train-only terminal-verdict
layouts and selects measured epochs by the weakest format/label margin, not just
average loss. It keeps the same sources and strict validation/audit thresholds.

## Limits

GPT-2 and its distilled student are related. Passing cannot establish transfer to
modern models, adversarial rewriting, mixed authorship, non-English text or a
particular person's writing. Pretraining overlap with journal text is unknown.
The same visible text can have different production histories. Reward logits are
not calibrated `ai_likelihood` values. Hidden node qualification is a separate
decision and may reject a source-audit pass.

## Licence

Original journal text keeps CC BY 3.0 or CC BY 4.0. GPT-2 retains its Modified MIT
notice. DistilGPT2 and ModernBERT retain Apache-2.0. Telegraph code retains MIT.
Our additions use `LicenseRef-zkasuran-SAND-1.0`. See LICENSE, NOTICE and
DATA-SOURCES.md. No upstream endorsement or human review is claimed.
