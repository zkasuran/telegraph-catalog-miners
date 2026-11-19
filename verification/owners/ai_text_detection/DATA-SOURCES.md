# Sources and rights

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
