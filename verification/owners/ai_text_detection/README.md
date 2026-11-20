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
points over the strongest length, overlap, answer-only or source-origin-length shortcut. Strict token
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

Layout-only continuation still failed bare validation and consumed no audit.
The next bounded job tests more source-backed training contexts: 127 eligible
non-overlapping windows from the same training articles. Synthetic counterparts
are unbuilt until the remote step runs. The original locked data stays intact,
with validation and audit excluded from augmentation. A larger actual training
corpus must pass every unchanged gate before any quality claim.

That job generated 126 valid source-backed views and used 124, producing 504
training pairs. The full data gate passed but the model still failed bare labels.
Those exact inputs are retained. The next fitting experiment weights the rare
bare training format symmetrically, without changing unweighted validation or
audit metrics. This is a testable improvement path, not a claimed quality pass.

Weight 16 improved bare validation to 76.56%, with JSON/prose each at 87.50%.
The model remained below the unchanged gates and consumed no audit. The next
bounded follow-up continues that exact improved checkpoint with weight 32 and
lower learning rate, keeping the same inputs and unweighted acceptance tests.

That follow-up did not improve the weakest validation margin, so the selector
kept the prior state and the audit stayed sealed. The next bounded experiment
tests 125 planned, further-disjoint training contexts from the same original
articles. They add no held-out rows or independent articles. Actual generated
counterparts and every source/data/quality gate still have to pass remotely.

## Current source-verdict experiment

Rank16 run44762535076 passed real expansion and gradient controls and selected
epoch0. Bare accuracy improved to82.81% with both labels above70%, but every format
still missed the stronger shortcut margin. No audit was consumed or registration
claimed. The current candidate directly resumes that selected adapter and its
paired readout at a fixed1e-5 rate. It keeps the same880 rows and all gates. This
is a bounded training hypothesis, not an unchanged binary retry or score mapping.

The source-verdict run44754728791 passed both data gates and actual reward-head
gradients but still failed strict quality. The current candidate retains its
rank8 checkpoint and expands only the supported trainable LoRA capacity to rank16.
Alpha/rank remains4, old slices/head are copied exactly and new B columns start
at zero. This is a remote training hypothesis, not a score map or a quality pass.
The same880 rows and all acceptance gates remain. See DATA-SOURCES.md for primary
capacity sources and the difference between retained slices and container bytes.

Compact run 44746932574 passed source, tensor and parser controls but still failed
strict verdict validation. The starting model was retained and the audit sealed.
The next job reuses the exact 128 supplementary provenance rows as constructed
JSON/prose verdict preferences, in addition to their existing origin supervision.
This directly trains the production reward head on 64 further licensed training
articles. It is not new generation, human review or exposed audit data.

All original 752 pairs and 12 bare rows remain unchanged. The existing 16
supplementary updates now include their 128 verdict pairs. Both the original
and combined 880-pair data gates run unchanged. Original and expanded integer
source-length baselines both enter quality margins. New gradients, full combined
surface probe and neural benefit remain remote-unverified. No registration is eligible.

## Limits

The margin-only run did not improve strict verdict quality and kept its starting
state. The compact training view used short, truthful opening/ending source quotes
instead of whole-passage cue copying. Original prompts, labels, bare rows and
held-out strings remained unchanged. Its full source-surface probe passed at 0.4723
but neural verdict quality still failed. No independent audit was consumed.

The supplementary-source run completed real generation and improved source-only
diagnostics, but the literal bare-verdict gate still failed. Its audit stayed
sealed. The next job reuses the exact generated data and tests a symmetric
training-only penalty on paired verdict-margin gaps. It does not rescale the
served score or lower gates. The new model's usefulness remains unverified until
the public CPU workflow and independent reference audit measure it.

Repeated fitting on the same sources has not passed the strict gate. The next
job adds 64 independently attributed training articles for source-origin
supervision only. Existing preference and held-out rows stay unchanged. It must
generate actual licensed counterparts, verify source/entity/byte separation and
report the extra source-only updates. The benefit is unverified until remote
measurement; no independent-audit or canonical result is claimed.

The length-controlled run reached 81.25% bare overall but still failed its
AI-generated label and source-length margins. The next continuation retains
both actual selected learned states and couples each existing source view's two
provenance counterparts under a symmetric harder-loss objective. It does not add
source rows, change verdict labels or alter the production executor. Its benefit
must be measured remotely under the same strict format, label and shortcut gates.

The correctness-objective run passed tensor, source and parser controls but did
not improve bare accuracy. Source-only checks then found a 78.125% validation
source-length shortcut missed by answer-length probes. It now enters every
quality margin. The next job tests equal-byte source-origin supervision through
a separate training-only encoder readout. It is not exported to the node package.
All original evaluator formats and thresholds remain, with the extra shortcut
making qualification stricter. No model improvement or independent audit is claimed.

The latest larger-corpus run generated 125 further views, used 248 total added
views and fitted 752 pairs across the same 128 training articles. Its data and
parser gates passed but the selected bare result remained 76.56%. No audit ran.
The next job reuses those exact actual inputs and tests a symmetric pointwise
correctness auxiliary beside the unchanged pairwise loss. It introduces no new
rows or runtime score mapping. Local gates verify policy and scalar controls;
actual tensor gradients and any model improvement must be measured remotely.

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
