# Data and model source grants

## [GOEMOTIONS-V4] Task-authoritative supervision

The first v4 job stopped before fitting at the primary dataset gate. Repeated
generic answers left only 11.5 percent unique negatives and 99.4 percent negative
overlap in the gate's split. The gate was not relaxed. Training now uses pairs of
independent original source excerpts as unrelated reference contexts, identical
on both sides. Reference categories are independent of task categories. Hashes
select sources but never appear in answers; IDs and padding do not supply diversity.
Bare training rows remain sparse. Generic and quoted validation/audit conditions
are unchanged and still test transfer away from training reference contexts.

Actual structural source checks now show 97.77 percent unique negatives, 0.52
percent top reuse, 3.09 percent split leakage and 50 percent length-only accuracy.
Those are non-neural checks. The primary learned surface probe and model quality
still run remotely. The failed job never fit a model and never scored the fresh
audit. Its source identities and bytes remain preserved.

Actual validation diagnostic run 44851670611 reproduced all 1,728 reference score
bits. On 32 deliberately contradictory prompt/excerpt comparisons, the task's
label was preferred in 18 JSON pairs, 19 prose pairs and 27 bare-label pairs.
Those are validation measurements, not a new independent audit or a node score.

V4 keeps the same pretrained assets and hyperparameters. It changes supervision:
training reasons do not copy the task. Validation uses generic reasons and a
second condition gives both answers the same unrelated excerpt. Excerpt labels are balanced independently of task labels,
including same-label controls. These preferences prioritize the derived category
of the task. They do not certify the truth of the excerpt or human reason review.

Every source/entity/thread/text identity from the exposed v2 and v3 audits is
quarantined. Only those exclusion keys are retained, not audit text for fitting.
The complete source inventory remains 211,225 rater records and 58,011 comments.
New disjoint groups are 576 training, 144 validation and 144 fresh audit groups.
Every group has three category contrasts in each of two conditions. Format views
share those groups. Training sees 3,456 pairs, not 3,456 independent sources.

Original GoEmotions annotations retain the Apache-2.0 grant below. All valence
mapping, compositions, explanations and preference construction are programmatic,
not human sentiment-preference review. The unchanged coefficients were fitted
only on original v3 training. No floating-point refit is needed to replay them.
The audit also measures an actual reference-engine answer-only shortcut.

Validation selects the weakest label across both conditions and every format.
The fresh audit cannot select an epoch. Every format and condition must pass
80 percent overall, 70 percent per label, five points over untrained and ten over
the strongest measured shortcut. A passing source audit is not a canonical win.

## [VALIDATION-DIAGNOSTIC] Frozen-model alignment controls

The development job reads the complete pinned GoEmotions source inventory and
reconstructs the original quarantined source grouping. Only existing validation
groups become score inputs. Original audit rows and node evaluation material do
not become development inputs. The exact validation-group and train-fitted baseline
hashes are in development-config.json. No coefficient is fitted again.

It selects eight groups per label with a fixed source-hash order, 32 groups total.
Each group has three contrasts in each of the original response formats. Conditions
are aligned tasks, no task prompt and a prompt from a different-label source group.
Answers remain identical across conditions. JSON/prose answers retain source
excerpts, so answer-only performance may reflect information in the answer.
Cross-label prompt results measure retention of the original preference, not
accuracy on a newly annotated task. All constructed preferences are programmatic,
not human sentiment-preference review. The original Apache-2.0 grant below applies.

This is validation development on an unchanged package, not an independent audit,
a new selected candidate or a registration-ready result. The original model and
its source-held-out audit remain frozen. Length, ties and immutable shortcuts are
reported alongside actual pinned reference-engine score bits.

## [UCI331] Sentiment Labelled Sentences

Kotzias, D. (2015). Sentiment Labelled Sentences. UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C57604
Source: https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences
License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
Archive SHA-256: afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343.

The reviewed granting clauses permit copying and adaptation for commercial use:

<!-- voice-gate:off -->
> copy and redistribute the material in any medium or format for any purpose, even commercially.
> remix, transform, and build upon the material for any purpose, even commercially.
<!-- voice-gate:on -->

Attribution, a license link and an indication of changes remain required.
No licensor endorsement is implied. Original text and labels keep their CC BY
grant. Dataset-derived statistics and frozen shortcut coefficients retain it.
Our computational code and competitive delta additions use SAND only for our
portions. SAND does not restrict the upstream text or annotations.

The complete relevant inventory is three original files, each with 1,000 physical
LF-delimited source records: Amazon, IMDb and Yelp. The importer reads every row
of each file before filtering. It does not treat a listing's first page as the
corpus. Selection uses the original fixed normalized group hashes and bounds
originals to 8 through 256 UTF-8 bytes. The original 768 training pairs and 128
validation pairs use Amazon/IMDb. All 256 audit originals use Yelp, with 128
Positive and 128 Negative labels. Cross-split normalized overlap must be zero.

Original source annotations are distinct from preference construction. JSON,
prose and bare-label answers and label flips were constructed by AI-assisted
code. They are not human-authored or human-reviewed preference pairs. Three
formats share the same 256 originals, not 768 independent examples.

The original model was selected on validation before the source-held-out result
was exposed. The frozen recovery changes only ignored registration metadata.
Every model field, tensor descriptor and payload byte must remain identical.
The original source-held-out audit is replayed solely for exported-runtime and
registration recovery. It is not used to select a later semantic candidate.
No hidden/private pool or development-exposed miner captures enter this data.

## [MODERNBERT] Foundation architecture and pretrained weights

ModernBERT by Answer.AI and LightOn. Apache-2.0.
Pinned base revision: 8949b909ec900327062f0ebf497f51aef5e6f0c8.
https://huggingface.co/answerdotai/ModernBERT-base/tree/8949b909ec900327062f0ebf497f51aef5e6f0c8

Apache-2.0 section 2 grants reproduction and preparation of derivative works
in source and object form, subject to its notice and attribution obligations.
The complete original Apache text and original notice are retained in
licences.json and in the original/recovered delta metadata. The pinned
foundation, engine and tokenizer hashes are in contracts.py and are verified
against actual downloaded files before execution.

## [KIT] Telegraph reference engine and package parser

Telegraph reference kit, Copyright (c) 2026 telegraphprotocol. MIT.
Pinned revision: 1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd.
https://github.com/telegraphprotocol/telegraph-wasm-baseline/tree/1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd

<!-- voice-gate:off -->
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction
<!-- voice-gate:on -->

The grant requires retaining its copyright and permission notice. Full notice
bytes are in licences.json and the original/recovered delta. Runtime jobs use
the unmodified CheckLoRA and real adapterexec built from this pinned source.
No foreign registered binary is wrapped or redistributed.

## Frozen model and evidence

Original training run 44266324976 at a3a6f86c1bd831b41de8d99d12b0620c3a287f6a.
Real exported-runtime run 44441475723 at a18b51ce4be30741a86e2432416b5b283a795b4d.
Original package SHA-256: f2eefcc7b7fd807bcbd260d97c8b3b0a6b50276199b475131df968a5b60d43d9.
The original artifact already embeds complete upstream and SAND terms.
Our recovery retains those bytes and adds one inert x-registration-recovery
field. SAND boundary is retained for failed checkpoints as well as exports.

Scope is short Positive/Negative English source reviews. Neutral, Mixed,
multilingual coverage, reason quality, backbone pretraining overlap and
population accuracy remain unverified. A source-held-out pass is not a node win.

## [GOEMOTIONS] New four-way training experiment

GoEmotions, Demszky et al. (2020), Google Research. Citation key:
demszky2020goemotions. Source revision c878a53ec1764bcb91a6178dc5c315202a3b5fa2.
https://github.com/google-research/google-research/tree/c878a53ec1764bcb91a6178dc5c315202a3b5fa2/goemotions
Original grant: Apache-2.0, repository LICENSE. The maintained dataset card names
that grant: https://huggingface.co/datasets/google-research-datasets/go_emotions

<!-- voice-gate:off -->
> each Contributor hereby grants to You a perpetual,
> worldwide, non-exclusive, no-charge, royalty-free, irrevocable
> copyright license to reproduce, prepare Derivative Works of,
> publicly display, publicly perform, sublicense, and distribute the
> Work and such Derivative Works in Source or Object form.
<!-- voice-gate:on -->

Section 4 requires the original licence and attribution to remain attached.
The original full LICENSE is verified by its SHA-256 and embedded in every new
delta and checkpoint package. SAND applies only to our competitive additions.

The importer checked all 211,225 rater rows across all three raw CSV files:
70,000 plus 70,000 plus 71,225. These contain 58,011 distinct comment IDs in the
actual files, not the README's rounded 58k. All source bytes are hash-pinned in
retrain-sources.json. Rater votes are aggregated by original ID with a strict
majority of at least two distinct raters. The original positive/negative emotion
mapping is retained. Neutral uses original Neutral annotations. Raw emotion
annotations are human-created; derived valence labels and preference pairs are
programmatic work, not human-reviewed sentiment preferences.

Mixed examples juxtapose one independently annotated positive comment and one
negative comment. Those composite texts and their labels are constructed, not
original human Mixed annotations. They test an emotion-valence proxy and do not
prove universal sentiment truth, natural Mixed performance or reason correctness.

The new inventory splits by original time before selection. Any author, thread
or normalized original crossing a time boundary is removed. Selected originals
also have distinct authors, threads and normalized texts within each split.
All original UCI groups are excluded from this new data. Frozen split sizes:
512 training groups, 128 validation groups and 128 new independent audit groups.
Each group has three wrong-label contrasts, not three independent sources.
The 1,536 training pairs and 384 validation pairs may select an epoch; the new
384 audit contrasts are scored only after the candidate package is frozen.

Prompt styles cover JSON input, raw text and text classification instructions.
Response audits cover JSON, prose and bare labels for all four allowed labels.
The strongest measured train-fitted length, overlap, keyword and answer-only
shortcuts remain reported. Coefficients were fitted once on training only and
are hash-pinned. Remote jobs verify their bytes rather than refit them.

This experiment warm-starts the existing licensed delta, then fits new tensors.
Registration 154 failed the real public quality floor. It is never resubmitted
unchanged. The old UCI audit is retained as history, not a fresh audit for this
candidate. No exposed node pairs or private pool rows are training inputs.
New neural quality and canonical qualification remain unverified until the real
bounded public CPU jobs and node decisions complete.

## [GOEMOTIONS-V3] Quarantined source continuation

Run 44704610971 selected a validation checkpoint but its independent audit failed:
75.78 percent JSON, 77.86 percent prose and 70.57 percent bare-label accuracy.
These measurements use 128 source groups and 384 contrasts per format. The exact
failed checkpoint keeps SHA-256 80729d67ac03086dfce4915252c665635c095e706c2332807a514f39ec0f72fd
and its complete embedded grants. It is not a passed release or a registration
candidate. Its actual source/audit evidence is preserved, never relabeled a pass.

V3 reuses that checkpoint only as licensed warm-start lineage. The 160 source
originals underlying its exposed audit are quarantined, along with their author,
thread and normalized-text identities. They cannot enter new fitting, validation
or a later independent audit. v3-quarantine.json contains only identity keys,
not raw comments or private pool rows. The old UCI originals remain excluded.

The complete raw inventory remains 211,225 rater rows and 58,011 source IDs.
V3 retains 11,892 bounded originals with unanimous agreement of at least three
distinct raters and unambiguous valence categories. The original emotion labels
are human-created upstream. The narrowed valence mapping is a programmatic proxy,
not a new human sentiment annotation. Prior-audit/old-UCI identity filtering
removes 79 of those eligible originals and the time/entity filter removes 478.

New disjoint groups: 768 training, 192 validation and 192 independent audit.
Each has three wrong-label contrasts. JSON, prose and bare responses share those
same originals, not independent populations. Some Positive/Negative/Neutral
examples combine two same-valence comments; Mixed combines opposite-valence
comments in both orders. Connectors vary across every label. All composed texts
and their preferences are constructed work, not human-reviewed sentiment truth.

Answers now use short source excerpts rather than repeat the whole prompt.
Numeric values in JSON are constructed category values, not human confidence
measurements. Reason correctness and confidence calibration remain unverified.
V3 validation measures every label in every format and selects the weakest-label
score first, then weakest format accuracy. The independent audit cannot select
another epoch after its result is exposed. All original quality gates remain.

Immutable train-fitted length, overlap, answer-only and prompt-length/composition
histogram baselines are measured on the new audit. Frozen coefficients and source
identities are bound in v3-config.json. Local source verification imports no
neural framework. Only bounded public CPU jobs may fit or execute the model.

## Passing v3 source audit and runtime artifact

Actual run 44723258067 at bb1eb538a4308956bf5b27eac952d5487c985de6 completed
training and the independent audit successfully. Model SHA-256:
ae2d3b0275e4f96419bee0e75396f445003afbfc401029353e5bc8ef9e3a061e.
Immutable actual report SHA-256:
fa0cb600acb55f4b8d5c597f1c8f9fe254b5084c66eab7d968037b508e54dbbd.

The audit uses 192 independent groups and 576 contrasts per format. The formats
share groups, not 1,728 independent originals. Actual reference accuracy:
87.3264 percent JSON, 89.5833 percent prose and 81.25 percent bare label.
The weakest per-label score is 74.3056 percent. Strongest train-fitted shortcut
accuracy is 59.5486 percent. Length, overlap, answer-only, prompt length and prompt
composition are all reported. The untrained comparison is an actual seed-matched
fresh PyTorch adapter, not native scores mislabeled as node reference results.

The report binds the real full parser, reference engine, three negative identity
controls and eight actual fixed vectors with identical repeat bits. All 3,456
actual reference scores were independently recomputed to the reported aggregates.
Audit did not select or change the model. Its exact bytes and source grants are
published unchanged. Serving uses the identical executor source at the audited
commit. Canonical qualification still requires a fresh matching node result.

## Endpoint lifecycle recovery

Registration 161 was rejected for an unreachable executor after its five-hour
restricted public lease ended. The actual runner log shows successful forwarding
then termination at the exact lease expiry, before delayed node qualification.
There is no new node quality score and no reason to fit the model again.

The recovery package adds one ignored registration identity field. It preserves
every v3 model field, tensor descriptor, tensor payload byte and original grant.
The prior source audit is replayed only to verify this fixed function's integration
and reference bits. It is not new training, a new independent audit or selection.
No private pool rows or development-exposed captures become new data.

The new deterministic lease guard is an owner-side process, not another agent.
It reads only exact public node/run/lease metadata and invokes installed actions.py
for gated continuation. It cannot publish, register, sign, cancel workflows or read
provider/signer keys. A changed owner attempt or terminal rejection stops it.
Each restricted lease stays five hours; each standard public CPU job stays within
350 minutes. The guard has a one-day ceiling and at most six continuation leases.
New keys are additive for the identical package and existing route. A healthy
endpoint is not replaced with a different model. Unknown writes are reconciled,
never blindly resent. The guard's live handoff proof still needs real remote runs.

## Completed fixed-function recovery verification

Actual run 44822282420 at ce2bb3f0f0423d26f6120149a5e8dd7d694631de completed
successfully. Immutable report SHA d9bd19c9b3e4f959df9ab41c653fd1448f4236c1322d45ca0e78b9636e231cbd.
Recovered model SHA 1d5c4c5f020dc75e3cd2c8b59bcd0c04a862ae6ab58f3031be9ad88b8778edd1.
All 3,456 actual reference replay scores equal their original v3 score bits.
The full parser, repeated fixed vectors, negative identity controls and literal
wire path passed. Original tensor fields and every source grant stay unchanged.

This verifies integration bytes, not a new independent source audit, new fitting
or canonical node quality. The origin source proxy and composition limitations
remain attached. A detached lease guard cannot be assumed persistent: it must
outlive its ephemeral owner unit through an approved deployment mechanism before
a fresh registration relies on it. No such lifetime proof is claimed here.
