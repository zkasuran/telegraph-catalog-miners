# Sources and rights

## Zero training dropout for paired comparisons

Run44806134085 passed source-margin value/teacher-detach/gradient and all data/parser
controls but still failed strict validation. The old selected state was retained
and the independent audit stayed sealed. No missing prediction-head export defect
was found: the primary foundation exporter includes frozen pooling, head.dense and
head.norm before the supplied scalar head, matching the sequence classifier.

The primary trainer defaults LoRA dropout to0.1 during training. [PEFT-LORA-0.21.2]
applies that dropout before each A projection, while validation uses eval. The
next fixed hypothesis sets only training LoRA dropout to0 to remove independently
sampled comparison noise. This is not a claimed cause or measured improvement.
All1008 data rows, state, source/labels, losses, rate and acceptance gates remain.
No runtime output or literal wire changes. The pinned base config has zero base,
embedding, classifier and MLP dropout and mean pooling; those settings stay fixed.

The public CPU runner must confirm no nonzero dropout module remains and repeated
training-mode scores on one actual training input match each other and eval before
an optimizer update. This native control is not a node/reference-engine result.
Actual quality and independent reference execution still gate registration.
comparison-dropout-primary.json binds source/config hashes. PEFT Apache-2.0 and
Telegraph MIT grants remain, with SAND only on our additions.

## Detached source-margin training consistency

Run44797766046 passed all original/combined data and task-head gradient controls
but retained the starting state after strict validation loss. Bare accuracy stayed
82.81%. The separate native source readout reached85.9375%, but this diagnostic is
not independent truth, candidate selection or a reference-engine score. No audit
was consumed. All failed artifacts and exact source/generation bytes remain.

The next training-only loss uses the same existing source annotation and learned
readout. For source logit z and recorded origin sign s it defines a detached target
`t = clamp(2 * Softplus(s*z), 0.25, 6)` and adds fixed0.1 times
`SmoothL1(chosen-rejected, t, beta=1)`. The positive floor means an incorrect teacher
cannot reverse the original preference. The upper bound limits training pressure.
The target is invariant to exchanging origin names and negating the source logit.
It is a fitted training aid, not a new source annotation or human preference.

Every existing pair, correctness, source-origin, counterpart and paired-margin
loss remains. No new row, source, answer string, runtime transform or threshold
change is introduced. Teacher gradients are detached only for this new term;
the original source BCE still trains that readout and adapted encoder. Existing
weighting, model, learning rate, source/entity/time splits and audit seal remain.
Actual tensor/gradient/refusal controls and model benefit run only remotely.

[TORCH-SOFTPLUS-2.15] The primary API defines beta1 Softplus as log(1+exp(x)) and
uses a linear branch above threshold20 for numerical stability. Existing
[TORCH-SMOOTHL1-2.15] defines the fixed beta1 consistency loss. Original PyTorch
grant/attribution remain in PYTORCH-LICENSE; no upstream implementation is bundled.
https://docs.pytorch.org/docs/2.15/generated/torch.nn.Softplus.html

## Fixed secondary minimal-verdict supervision

Run44789054038 completed all64 source groups, retained the exact39-group prefix
and passed25 actual token-prefix controls. Its generation bytes are now immutable:
f4d8c1928e27c666d0255172604e756556696bfc2827d55e852530507f185f96.
All three data gates and the package parser passed, but strict validation quality
still failed and no independent audit was opened. Native aggregates are not wins.

The next job restores that completed generation and original producer without any
new neural generation. All880 earlier preferences remain unchanged. Two whole
secondary training article groups, chosen by lowest group hashes before fitting,
supply four additional literal bare-label comparisons. The other124 secondary
rows retain truthful JSON/prose quotes. Total1008 preferences now include16 bare
rows; unchanged uniqueness, repeated-negative and internal leakage gates still
apply. A separate worst-case any-split leakage bound must stay at or below0.10.

The secondary phase uses the existing symmetric bare weight16, normalized by its
actual mean. Both source origins have equal total weight. Original training and
the first supplementary phase are unchanged. The learning rate, losses, model
state, all source/entity/time splits and every quality/runtime gate remain fixed.
This is additional minimal-wire supervision, not answer padding, an origin prior
or a score transform. Generated text/source rights are retained verbatim; these
constructed preferences are not human-reviewed annotations. Model benefit remains
unverified until the bounded CPU job and independent reference audit measure it.

## Retained successful generation with a fixed token extension

Run44784633424 failed the unchanged24-word excerpt gate during secondary GPT-2
generation, before fitting. Its actual partial artifact retained39 complete source
groups,78 records and112299 bytes, SHA-256
560aa8bc4a15593b1819f8fe06e05a53a0f9851ddf56edc235c9f593bfff438d.
The old880 rows, selected checkpoints and every quality gate remain unchanged.

The next job reuses those39 groups verbatim. Only25 missing groups receive a fixed
128-token generation budget with the same predetermined source seed. A real
same-job64-token control must match the first64 tokens of the128-token completion.
No source skipping, seed search, model-score selection or reduced word/byte gate
is allowed. The originally rejected draw was not journaled, so its historical
token identity is unverified; the new control measures the current pinned job.

Raw generated candidates now carry source/grant/generator identity in a journal
before excerpt acceptance. A journaled candidate is not an accepted preference
or human-reviewed material. Both successful and failed output retain source and
model grants. New completion, fitting and reference quality remain unverified.

[TRANSFORMERS-GENERATION-5.22.1] The primary GenerationConfig defines
min_new_tokens and max_new_tokens in tokens, not words. This explains the actual
refusal without treating64 tokens as24 words. The original Apache-2.0 grant
remains attached; only APIs are used and no upstream implementation is bundled.
https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/generation/configuration_utils.py

## A further independent training-source block

Run44771475621 restored the selected rank16 state exactly and passed the data and
parser gates, but both lower-rate fitting epochs failed to improve strict quality.
The same selected state was retained and the independent audit stayed sealed.
No unchanged failed binary is registered for another draw.

A fixed source-only scan of1611 candidates selected64 further2012-2014 training
articles. The full pinned inventory remains32141 XMLs/19701 article IDs, with a
complete4555-entry legacy v1 candidate pool. This bounded scan is not the corpus.
Every original/retired source, named author and normalized text from both previous
source blocks is excluded. Individual original Git blobs and CC BY grant clauses
are retained in secondary-training-sources.json and reverified before generation.

The old880 preferences and all their source/producer hashes remain unchanged.
Only bounded public CPU work generates the new128 counterpart rows with fixed
GPT-2 seeds. New generation is unbuilt until that actual step succeeds. Training
then has1008 preferences in256 source groups, with94 original,16 first-supplement
and16 secondary updates per epoch. Original752, previous880 and combined1008 data
gates all run unchanged. No new bare answer strings, audit data or runtime score
transform are added. The old source-length comparator remains in every margin.

expanded-training-source-bundle.json binds the unchanged parent bundle and new
source manifest. Reports, packages, audit freeze and failed checkpoints use the
same full identity. Attribution in tensor headers retains exact original clauses,
authors, URLs and hashes without duplicating paragraph text. Complete originals
remain in source sidecars. Constructed preferences are not human-reviewed votes.
New generator/model results and independent runtime quality remain unverified.

## Exact selected rank16 lower-rate resume

Run 44762535076 verified all88 original matrix deltas with zero actual difference,
retained old slices/head and finite gradients in all88 added B matrices. Its actual
selected epoch0 reached bare82.81% (Human93.75%, AI71.88%), JSON84.38% and prose87.50%.
Every format still missed the strict78.125% source-length baseline plus ten points.
Later epochs degraded; no independent audit was opened. A partial improvement is
not registration eligibility or a canonical win.

The next candidate resumes the exact selected rank16 adapter and its paired saved
origin readout, not the older rank8 checkpoint. Original producer, selection,
source and package evidence are retained beside them. The adapter SHA-256 is
05d0d605cbe0e4f05af43f0e15562fbba6d3d5b41d45846b38b495f51b230b6f;
the readout SHA-256 is47ae66f72ff3d6b984d9550b8068a84611f84dabe7797fd1d6e5afcb9a51b1ab.
No tensor payload is interpreted locally. Original grants and attribution remain
attached to these failed-but-licensed checkpoints. The readout remains unexported.

Fresh bounded AdamW uses a fixed1e-5 rate, one-third the prior3e-5 rate, because
later updates worsened the weakest measured validation margin. This hypothesis
does not use audit data. Four epochs, all880 rows, all losses and integer baselines
remain unchanged. The runner requires exact direct tensor equality before new
fitting. It does not re-expand, reroll generation or rescale the served score.
Neural benefit and independent reference qualification remain unverified.

## Retained rank16 capacity continuation

Run 44754728791 passed both the original752 and combined880 data gates. Its
combined surface probe was0.4992. Actual supplementary gradients reached two
reward-head tensors and176 LoRA tensors. Both fitting epochs still failed strict
verdict quality and the starting model was retained. No audit was consumed.

The next candidate expands the same learned rank8 state to the current foundation
maximum rank16. Alpha changes from32 to64 so alpha/rank remains4. Every learned
rank8 matrix slice and reward-head tensor is retained. Only the extra A rows use
the pinned primary seeded initialization; extra B columns are zero before fitting.
This preserves the initial delta mathematically while adding trainable directions.
It is not a runtime score transform or a claim of identical larger-container bytes.

[PEFT-LORA-0.21.2] The primary LoraLayer source sets vanilla scaling to
`lora_alpha / r`, initializes A with Kaiming uniform and initializes B to zero.
Its original Apache-2.0 header and complete grant remain applicable. We call the
same APIs and bundle no upstream implementation. The source hash is recorded in
capacity-primary-sources.json.
https://github.com/huggingface/peft/blob/v0.21.2/src/peft/tuners/lora/layer.py

[TELEGRAPH-LORA-FORMAT] The pinned package format declares R_MAX=16 and feeds the
engine alpha/rank. [TELEGRAPH-LORA-PARSER] The actual Go parser also declares
LoRARMax=16 and refuses a rank above the foundation's RMax. Both original MIT
grants and source hashes remain retained. The foundation, tokenizer and literal
wire do not change.

The runner verifies actual retained tensor slices, unchanged head bits, zero new
B columns and every initial FP32 matrix delta within a fixed absolute1e-6 bound.
This tolerance is for expansion arithmetic, not runtime acceptance or quality.
Actual new-column gradients and six tensor refusals run only remotely. All880
data rows, old checkpoints, losses, source/entity/time splits, integer baselines,
CPU/epoch ceilings and strict validation/audit/runtime gates remain unchanged.
Neural benefit and independent reference results are unverified until measured.

## Independently sourced training verdict preferences

Compact run 44746932574 passed the unchanged full source-surface probe at 0.4723,
all tensor controls and the full parser. It still failed strict JSON and bare
quality gates, kept the starting state and consumed no independent audit.

The next job adds verdict supervision on the exact 128 previously generated
training-origin rows from 64 supplementary articles. Their original data hash is
8b56251827c93deb1723e2db9ddc26c0140667fa1cd0c7dfe0d7da3717c7dc2a,
producer run 44687535825. There is no new generation, source selection or audit use.
Per-article CC BY grants and recorded GPT-2 origin remain attached. Equal-byte
prefix transformations are marked in the source contract. The new preferences
are constructed from those provenance labels, not human-reviewed votes.

Original 752 preferences, 12 bare rows and 94 original updates remain intact.
The existing 16 supplementary updates now train both origin and verdict losses
on 128 additional JSON/prose pairs. The combined population is 880 preferences
in 192 training article groups. Both original and combined populations must pass
the unchanged upstream repetition/surface gates. No repeated bare string is added.
Full tokenization, reward-head gradient controls and model benefit run remotely.

Both the original train-fitted source-length rule and the expanded train-fitted
rule enter every strongest-shortcut margin. The original 78.125% validation
comparator cannot disappear when training material grows. No score transformation,
quality-floor reduction or changed held-out wire is introduced. The independent
audit stays sealed until every strict validation format passes.

## Compact source-grounded training cues

Run 44695379185 passed its actual margin tensor controls and package parser but
did not improve the strict verdict result. It kept the starting checkpoint and
consumed no independent audit. No failed checkpoint is re-registered unchanged.

The next training layout quotes four opening words and four ending words from
each actual source passage instead of repeating the whole passage in the answer.
Both preference sides have the same truthful quotes and opposite verdict labels.
There are no case IDs, padding strings or invented origin explanations. Original
prompts, source text IDs, article/entity/time roles and provenance remain intact.
Bare training rows are unchanged. Held-out JSON, prose and bare responses keep
their exact original wire strings.

The data-only smoke retains 752 pairs and 12 bare rows. Unique rejected fraction
is 0.9867021276595744, largest repeated negative share 0.007978723404255319 and
internal negative leakage 0.008928571428571428, within the unchanged thresholds.
The full upstream surface probe and model-quality gates still run remotely.
Shorter source quotes are a training-format hypothesis, not a measured neural
improvement. All original grants and quote attribution remain in the source bundle.

## Paired verdict-margin alignment

Run 44687535825 produced all 64 supplementary groups and 128 provenance rows.
Original source/byte/overlap and tensor/checkpoint/parser controls passed. The
native source readout reached 82.8125% on validation, but literal bare verdicts
still failed per-origin and shortcut margins. No independent audit was consumed.
The exact actual supplementary output is retained by byte hash, producer run and
source-bundle identity. Restoration is not reported as new generation.

The next fixed train-only objective adds coefficient 0.1 times SmoothL1 between
the two correct-verdict margins in each existing source view. This is symmetric
under swapping source origins. It discourages one verdict margin from dominating
even when pair losses are small. It does not change runtime scores, source rows,
preference labels, selection rules or quality gates. Improved transfer is a
hypothesis that requires the actual remote result, not an assumed causal finding.

[TORCH-SMOOTHL1-2.15] The reviewed primary implementation uses a quadratic term
for absolute error below beta and an L1 term otherwise. Beta is fixed at 1.0.
Value/gradient and shape/nonfinite/dtype refusal controls run in the public CPU
job before fitting. Original PyTorch terms remain in PYTORCH-LICENSE; only its
API is called and no upstream implementation is bundled.
https://github.com/pytorch/pytorch/blob/v2.15.0/torch/nn/modules/loss.py#L988
https://docs.pytorch.org/docs/2.15/generated/torch.nn.SmoothL1Loss.html

## Independent source breadth extension

The coupled run 44675121682 passed its real tensor, checkpoint and parser checks
but did not improve the strict origin-verdict quality result. The original 752
preference pairs and every validation/audit source remain intact.

A fixed scan of 1,259 source candidates found 64 additional training articles.
The complete pinned inventory still contains 32,141 XMLs and 19,701 article IDs;
the complete legacy v1 candidate pool contains 4,555 entries. These are different
counts. New sources have 2012-2014 dates, verified original Git blobs and individual
CC BY grants. Existing and retired source IDs, named authors and normalized
source text are excluded. Original clauses, DOI, source URLs and transformations
are retained in extra-training-sources.json, not replaced by a blanket grant.

In the original breadth experiment the bounded public CPU workflow generated GPT-2 counterparts.
They provided 128 source-provenance supervision rows, not literal preference pairs
or human-reviewed votes. Source inputs use equal-byte UTF-8 prefixes and normalized
overlap refusals. Generation has fixed seeds, retained provenance and no score
selection or rerolls. Those actual generated bytes are retained with their producer.

The additional source-only phase has 16 explicitly reported optimizer updates per
epoch. The original preference phase still uses 752 rows and 94 updates. No old
pair-data criterion or quality floor is weakened. Validation and audit remain
unweighted and the train-fitted 78.125% source-length validation shortcut stays.

training-source-bundle.json binds the unchanged original source manifest and the
supplement's bytes. Package and independent-audit source identities use this
composite hash. Original producer checks retain the base manifest identity. All
source text, source annotations and constructed preferences keep their distinct
provenance and licences, including in failed checkpoint exports.

## Symmetric counterpart continuation

Run 44666934403 produced a real bare-format improvement to 81.25%, but its
AI-generated label accuracy was only 65.62% and its strongest source-length
shortcut was 78.125%. The strict gate refused it and the audit stayed sealed.
Its encoder-gradient and head-exclusion controls passed. Both selected learned
states, production adapter and separate source-origin readout, are retained.

The next objective couples the same two provenance counterparts within each
existing source view. For their existing training losses a and b it uses
`(a+b)/2 + 0.25*abs(a-b)`. This is 75% of the larger loss plus 25% of the smaller,
so it is invariant to origin-name swaps. It is not an AI-specific weight, changed
preference label or runtime score transformation. It gives less opportunity to
ignore a systematically harder counterpart while keeping both source roles.

All 752 literal pairs and their original source/entity/time roles remain. A
loader batch is one source view containing two literal rows. Four view batches
still yield one eight-row optimizer update and 94 updates per epoch. These actual
batch semantics are recorded separately from the reference hyperparameters.
Validation and audit stay unweighted and every existing gate remains active.

[TORCH-ABS-2.15] The pinned primary API computes the absolute value of each tensor
element. New tensor value/gradient controls test symmetric, unequal and tied
counterpart losses in the actual CPU runner before fitting. This API use retains
PyTorch's original grant in PYTORCH-LICENSE. No new third-party implementation,
model or data source is bundled by this change.
https://docs.pytorch.org/docs/2.15/generated/torch.abs.html

## Length-controlled source-origin supervision

Run 44660532017 passed actual tensor controls, data gates and the full package
parser. Its selected epoch reduced validation loss but bare accuracy stayed at
76.56% (Human 84.38%, AI-generated 68.75%). No independent audit was consumed.
The exact source-fitted checkpoint is retained, not registered.

A source-only measurement found that a train-fitted integer source-byte threshold
gets 566/752 training origins and 50/64 validation origins correct. The latter
is 78.125%. Prior answer-length probes did not measure this input-text shortcut.
The next frozen baseline includes it in every validation and audit format's
strongest-shortcut margin. No existing 80%, 70%, five-point or ten-point floor is
lowered. The unchanged upstream pair-data gate continues to measure its original
answer-surface criteria; this extra source-origin rule strengthens quality gates.

The new auxiliary uses the same accepted 752 training passages from 128 source
articles. It selects a contiguous UTF-8-boundary prefix with equal byte count
within each paired source view. No score, origin-specific choice or retry selects
the cut. Prefix changes and original normalized text IDs remain bound. Normalized
train/validation crop overlap is zero and the auxiliary's integer source-length
baseline is 50% on both splits. Original evaluator and audit inputs are unchanged.

Source provenance supplies the auxiliary targets, not human preference review.
A separate training-only linear readout uses the adapted encoder's CLS state.
Its fixed coefficient is 0.25 and its smoothed human/synthetic targets are 0.9/0.1.
The readout is not attached to the production model and is excluded from its
178 learned adapter tensors and node package. Only shared LoRA features learn
from this additional supervision. Native readout validation is diagnostic only,
never a candidate-selection or independent-audit metric.

[MODERNBERT-5.22.1] The pinned primary implementation returns
`BaseModelOutput(last_hidden_state=hidden_states)` from `ModernBertModel.forward`.
The sequence classifier separately pools the encoder state and applies its
prediction head and classifier. This permits a training-only readout without
changing the node's classifier or package contract.
https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/modernbert/modeling_modernbert.py#L401

[PEFT-BASE-0.21.2] `get_base_model` returns the underlying adapted model for a
non-prompt-learning adapter. The primary implementation and original Apache-2.0
headers are hash-bound in origin-primary-sources.json. API calls do not bundle
either source implementation. Apache section 2 grants reproduction and preparation
of derivative works; the complete terms and attribution remain in sources.json.
https://github.com/huggingface/peft/blob/v0.21.2/src/peft/peft_model.py#L1079

## Current frozen correctness-objective experiment

Run 44646831194 generated 125 valid third-context views without structural
refusals. Combined fixed-order trimming kept 248 added views, giving 752 fit pairs
in the same 128 articles. Its data and full-parser gates passed. The selected
checkpoint still scored 76.56% bare overall and 68.75% for AI-generated, so the
independent audit stayed sealed. These validation figures are source development,
not an independent audit or canonical result.

The next job restores the exact third-context data SHA-256
`7c42ad4c1b08f4029c1ec2168431bb93f3cdd67b1bf15d86bfb701dda3937574`
with its actual producer, source and plan bindings. It does not regenerate or
select new examples. All original training and held-out roles remain fixed.

[TORCH-BCE-2.15] PyTorch v2.15.0 `BCEWithLogitsLoss` source lines 719-844
defines a stable binary loss on raw logits, mean reduction and same-shape targets
in [0,1]. Its forward method calls `binary_cross_entropy_with_logits`.
https://github.com/pytorch/pytorch/blob/v2.15.0/torch/nn/modules/loss.py#L719
https://docs.pytorch.org/docs/2.15/generated/torch.nn.BCEWithLogitsLoss.html

The fixed train-only objective retains [TRAINER-1ed063] pairwise loss and adds
0.5 times the mean pointwise correctness loss on those same chosen/rejected
logits. Targets are 0.9 and 0.1 for correct and incorrect verdicts, equally for
both origins. No extra training rows, origin weighting, runtime sigmoid or
score transformation is introduced. This constrains a common reward offset that
the pairwise difference alone does not constrain. Improved discrimination is a
remote hypothesis, not a measured result. Targets are not authorship probabilities.

[TORCH-LICENSE-2.15] The reviewed PyTorch grant states: "Redistribution and use
in source and binary forms, with or without modification, are permitted provided
that the following conditions are met:". Copyright, conditions and disclaimer
are retained in `PYTORCH-LICENSE`. Only the API is called; PyTorch source and
weights are not bundled. Its dependency package retains its own grant.
https://github.com/pytorch/pytorch/blob/v2.15.0/LICENSE

## Journal controls

`elifesciences/elife-article-xml`, revision
`c3cd502f94edfb5aa0c5870ed9eff053773c6125`, article tree
`4ee79e98a7dbdcd3dd040be4b836e0843520c1fd`.

Each selected article's XML contains a CC BY application clause permitting
unrestricted use and redistribution provided that the original author and source
are credited. The exact original clause, licence URI, author names, DOI, version,
Git blob and SHA-256 are retained per article in sources.json. Grants differ by
version. No repository-wide blanket grant is assumed.

CC BY 3.0 section 3(b) grants the right "to create and Reproduce Adaptations".
CC BY 4.0 section 2(a)(1) grants reproduction, sharing and adaptation. Attribution,
change identification and retained source rights apply. The full terms are bound
in sources.json. We remove XML markup, collapse whitespace and select a contiguous
24-word abstract excerpt. Figures, bibliography, linked citations and attributed
quoted blocks do not enter the pilot.

## Generators

GPT-2, `openai-community/gpt2`, revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`. OpenAI's Modified MIT terms grant the
right "to deal in the Software without restriction". The output statement says
the creator's content is theirs to use. The disclosure request is retained:
"We only ask that you use GPT-2 responsibly and clearly indicate your content was
created using GPT-2." The complete notice is embedded in every successful export
and every preserved checkpoint. Generated fragments carry source citation and
source rights when they reproduce licensed source material.

DistilGPT2, `distilbert/distilgpt2`, revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`. Its pinned model card declares
Apache-2.0. Apache section 2 grants a copyright licence to reproduce and prepare
derivative works. The full grant and section 4 conditions are retained. This is
a distilled GPT-2 checkpoint, not an independent generator family. No OpenWebText
or WebText corpus is downloaded or redistributed.

## Reward foundation and reference code

ModernBERT, `answerdotai/ModernBERT-base`, revision
`8949b909ec900327062f0ebf497f51aef5e6f0c8`. Its model card says the architecture,
weights and training code are released under Apache 2.0. Section 2 grants a
copyright licence to reproduce and prepare derivative works. Section 4 conditions
and original attribution remain attached.

Telegraph reference kit, revision `1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd`,
Copyright 2026 telegraphprotocol. MIT grants permission "to deal in the Software
without restriction" subject to retaining the copyright and permission notice.
The pipeline downloads this pinned source archive remotely and invokes its
unchanged trainer, exporter, full package parser and reference executor.

## Exclusions and preference provenance

HC3 is excluded. Its share-alike obligations and source-specific restrictions do
not become SAND merely because its card names a licence. No hidden evaluation-pool
row, provider response capture or prior exposed audit is a training input.

Journal text is source text. Publication metadata is source annotation. Generated
origin records are machine provenance. Verdict comparisons are constructed
preferences. None of those records is described as human-reviewed preference data.
The source manifest stays byte-identical across the checkout integration repair.
Its separate fit, validation and audit groups are not rebuilt after a failed job.

## Continuation revision and independent audit

The first model's validation result was 0.5469, below the unchanged 0.80 gate.
Its original 64 audit articles and named authors are quarantined. A new fixed
hash-order scan of 151 source candidates yielded 64 fresh eligible audit groups.
They are disjoint from fit, validation and retired audit source/entity groups.
The complete original inventory still supplies the population count. The scan
of 151 is a bounded refresh, not the size of the source corpus.

Training and validation articles are unchanged. The model continues from the
exact source-fitted checkpoint with a declared optimizer reset. No old audit
output supplies training labels or tunes the candidate. The new audit has no
model quality score yet and stays sealed until all validation formats pass.

Audit views collapse whitespace on both origins and select the largest shared
whole-word prefix from 16 through 24 words that fits both unchanged 256-byte
slots. Word count is equal within each source pair. No label, model score or
generation retry selects a view. Every original source span and generation
record remains attributed. Byte and identical-text refusal gates remain active.

[PEFT-STATE-0.21.2] The pinned primary implementation documents that
`get_peft_model_state_dict` and `set_peft_model_state_dict` handle the arbitrary
adapter-name removal and insertion when preserving learned state. The pipeline
checks all learned keys, shapes, finiteness and exact tensor equality after
loading, before any continuation fitting. It does not claim to restore an
optimizer that the original trainer never exported.
https://github.com/huggingface/peft/blob/v0.21.2/src/peft/utils/save_and_load.py

## Locked development reuse and repeated-negative repair

Run 44573838929 preserved the actual generated training and validation data.
Its SHA-256 is `032c08d6b59fe66c22cc498e6ac6b9f7464676443f140469e22b31142e5cd3e3`,
identical to the original run's data. The new job restores these exact existing
bytes, checks every source ID and original split role then records the actual
producer separately from the current preparation run. It does not claim that
restoring a file is a new generator operation.

The prior four bare-format article groups passed repetition share checks but
leaked four repeated stock negatives into 38 internal holdout rows: 10.53%, above
the unchanged 10% gate. The repair uses one whole training article group, giving
one distinct opposite bare string per origin. All 256 train pairs and 64
validation records remain. The primary split expression is retained verbatim.
This deterministic construction removes reuse for every internal split, not by
selecting a favourable seed. The full upstream surface probe still runs remotely.

## Format-resilience continuation

Run 44588237548 passed its full data and parser gates. Selected validation was
82.81% JSON, 89.06% prose and 71.88% bare labels. Bare Human accuracy was 68.75%,
so the unchanged all-format/per-label gate refused it. Its exact licensed
checkpoint is retained for the next continuation. No audit result selected it.

The next training view mixes symmetric verdict-first and verdict-last JSON/prose
responses by fixed source hash. Source text, preference labels and both sides'
format are unchanged. The terminal verdict layout teaches the relation used by
bare responses without introducing repeated bare negatives. Validation and audit
keep their original literal formats. No validation row is used for fitting.

[TRAINER-1ed063] The guarded training loop retains the primary Bradley-Terry loss,
collator, AdamW, clipped updates and bounded scheduler. Epoch selection uses actual
validation gate pass and the weakest format/label margin before measured mean loss.
It never substitutes a synthetic loss or reads audit data. Full primary MIT terms
remain in sources.json and checkpoint metadata.
https://github.com/telegraphprotocol/telegraph-wasm-baseline/blob/1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd/float_runtime/train_adapter.py

## Additional train-only source windows

Layout-only continuation did not clear bare-label validation. Its full data and
parser gates passed and its audit remained sealed. More distinct training
contexts are the next source-bound build path, not a lower acceptance gate.

A source-only scan found 127 eligible second windows in the existing 128 training
articles. Original inputs use words 0-11 as generator prefix and words 12-35 as
the human control. New windows use words 36-47 and 48-71. Exact character spans,
DOI, original article SHA-256, source URL, authors and CC BY grant are retained.
The 127 windows are additional views, not independent articles. No validation or
audit article enters this plan.

Only bounded public CPU work generates new counterparts with pinned GPT-2.
Structural byte, token, source and normalized-identity checks can refuse a view.
Refusals are retained without using a reward score or retrying a seed. At least
80 valid views are required. Whole views are trimmed in fixed order only to keep
complete optimizer blocks, with exact used IDs in the frozen report.

The larger actual corpus can include four original bare article pairs while
passing the unchanged repetition gate and a worst-case negative reuse bound of
10% for any internal split. The real upstream surface probe and quality tests
still run. No generated source-window output is claimed before the job executes.
All original human-control assumptions, generated-text disclosures and source
rights remain in force. HC3 remains excluded.

## Reused actual training augmentation and rare-format fitting

Run 44622803647 produced 126 valid additional views and refused one structurally
invalid view. Fixed complete-block trimming used 124 views, giving 504 train
pairs in 128 original article groups. The full upstream data gate passed:
negative leakage 0.0267, length shortcut 0.5 and source-surface probe 0.516.
Its original source and producer hashes are retained with exact generated bytes.
Restoring those bytes is preparation, not a new model generation claim.

The selected model still failed bare-label validation, so the independent audit
was not consumed. The next training-only experiment gives the rare bare-format
pairs weight 16, equally for both origins and pair sides. Weights are normalized
by the actual mean training weight. Both weighted objective loss and unweighted
pair loss are reported. Validation, audit, baselines and acceptance thresholds
remain unweighted and unchanged. No validation/audit row chooses a new training
example or supplies fitted coefficients.

## Bounded follow-up from an actual improved checkpoint

Run 44632309433 applied the declared symmetric weight-16 policy and preserved an
actual improvement: JSON/prose validation 87.50%, bare labels 76.56% (Human
84.38%, AI-generated 68.75%). The same gate refused it and consumed no audit.
Its licensed checkpoint SHA-256 is
`39c3b89efa9da2c61ab841d82a7ceb31ba2da1acafbf6d2710163a987a073efe`.

The next continuation uses that checkpoint with weight 32, within the already
bounded training policy, learning rate 0.00004 and at most four epochs. Input
sources and actual generated bytes are identical. Weights remain equal across
origins and pair sides and are normalized by their actual mean. Validation and
audit stay unweighted. No score rescaling, new pool material or relaxed threshold
is used to turn this source development result into a qualification claim.

## Further disjoint training contexts

The weight-32 run did not improve the retained checkpoint's weakest validation
margin and consumed no audit. A source-only scan found 125 further eligible
windows in the original training articles: words 72-83 as generator prefix and
84-107 as the human-source context. These do not overlap the original words 0-35
or earlier words 36-71. They are new views, not new independent articles or
validation/test captures. Full source attribution and exact spans stay in the plan.

Actual generation is bounded to 128 public CPU calls with the same pinned GPT-2
and source checks. At least 84 valid additional views are required. The larger
actual corpus may expose six original bare article pairs only if the unchanged
upstream data gate and the same worst-case 10% negative-reuse bound pass. Source
independence stays 128 training articles. Validation/audit sources, formats and
every quality threshold remain unchanged. The training recipe uses symmetric
weight 16, lr 0.00003 and at most four epochs from the retained actual checkpoint.
Synthetic counterparts are unbuilt until the remote generation step executes.
