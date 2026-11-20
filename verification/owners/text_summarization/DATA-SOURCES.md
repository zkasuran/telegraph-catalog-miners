# Source grants and task scope

## eLife articles

The source is the eLife article XML repository at revision
`c3cd502f94edfb5aa0c5870ed9eff053773c6125`. The complete flat article subtree
`4ee79e98a7dbdcd3dd040be4b836e0843520c1fd` contains 32,142 entries and
19,699 v1 XML files. The complete inventory was checked, not its first page.
13,312 articles were verified during candidate collection. The resulting
freeze selects 448 articles in 224 conservative author/ORCID/DOI components.
It is not claimed to be a census of eligible articles.

The selected articles explicitly grant CC BY 4.0. One retained clause reads:

> This article is distributed under the terms of the Creative Commons Attribution License permitting unrestricted use and redistribution provided that the original author and source are credited.

Each article has its exact clause, authors, immutable URL, original XML SHA-256
and Git blob hash in `data/manifest.json`. The original grant controls the source
text. The complete CC BY legal code is in `CC-BY-4.0.txt`.

The complete substantive abstract and unchanged published title are extracted
by whitespace normalization. External references and quoted third-party passages
are excluded. Published titles are human-authored source associations, not human
preference votes. Pair construction is AI-assisted. Each reciprocal task asks
for a title for Study A or Study B with both complete abstracts present.
This tests target-study title relevance. It does not establish full summary
faithfulness, independent scientific truth or general factual correctness.

## ModernBERT and Telegraph

ModernBERT-base at `8949b909ec900327062f0ebf497f51aef5e6f0c8` is supplied
under Apache-2.0. Its retained grant says:

> each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work and such Derivative Works in Source or Object form.

The MIT Telegraph kit at `1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd` says:

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software

Verbatim grant punctuation above is retained. Full terms are in
`UPSTREAM-APACHE-2.0.txt` and `KIT-LICENSE`. `reference-pins.json` binds
the kit archive and each base-model file. Runtime foundation, engine and
tokenizer hashes are pinned in `runtime_check.py`.

## Freeze and evaluation

Revision 2 retains the original 128 training and 32 validation components. It
replaces the scored original audit with 64 fresh independent source components.
The old audit SHA-256 is listed in `exposed-audit.json` and its source/entity
identities are excluded from all new fitting and evaluation. Source text still
has explicit CC BY 4.0 grants. The source-year range now admits 2014-2026. A
rescan of the same 13,312 cached files produced 4,007 eligible unique records.
The complete 19,699-file inventory was checked. No exhausted eligible-corpus
claim is made. Fresh audit selection found 146 independent candidate components
after excluding old sources/entities and selected 64 without model scoring.

The failed revision 2 recipe used learning rate 0.001, gradient accumulation 4 and
zero LoRA dropout, with the same three epochs, rank 8, alpha 32 and seed 42.
This gives 192 updates instead of 48. Selection still uses validation loss alone.
All development formats must clear the unchanged quality and baseline gains
before any new audit score is computed. A failed development candidate leaves
the fresh audit unscored. Original source grants and numerical baseline
coefficients are unchanged.

The actual revision 3 run 44778080472 failed the same development gates at
33/64 in every format and left the clean audit unscored. A matched batching
diagnostic now uses four original training groups and their unchanged complete
two-reference tasks. Stochastic and reciprocal A/B batches have equal size,
learning rate, update count and per-row exposure. This adds no source text or
constructed preference label. It cannot emit registration readiness. Full
original grants apply to every retained diagnostic artifact.

Training-only diagnostics select four complete existing training groups with
their original titles. One control retains the original two-reference task;
another asks about one full requested abstract. No source sentence or title is
cropped. Neither validation nor audit rows are fitted or scored by this mode.
Tiny-group fit is a representation/gradient control, not general capability.

Revision 3 constructs a single requested-reference warm-up only from all 256
exact original training rows. Published titles and complete substantive abstracts
remain unchanged. One epoch uses learning rate 0.0005 with 64 updates. A fresh
optimizer then runs the original two-reference task for three epochs at 0.0001
with 192 updates. No new source, preference annotation or evaluated audit row is
introduced. Warm-up is fitted material, not an independent evaluation. The same
development gates protect the still-unscored audit. Numerical baseline coefficients
and source split hashes remain unchanged.

Train has 128 components and 256 reciprocal pairs. Validation has 32 components
and 64 pairs. Audit has 64 components and 128 pairs. Source/entity identities
and normalized title/abstract text do not overlap across these splits.
Complete constructed-task availability is strictly ordered by time. This is
not a new-topic holdout. All raw/JSON/prose task-answer inputs fit 256 tokens.

Analytical coefficient vocabulary, IDF, answer-only log odds and directions
were fitted once using train only. They are immutable. Validation loss selects
one checkpoint after all three fixed epochs. The independent audit is used
only after restoring that checkpoint and is never used for later tuning.
Source integrity and token-length checking do not score the audit.

Length, token length, TF-IDF overlap and answer-only count controls replay at
50% on development in each format. The upstream answer-only surface probe
uses an internal pair-level diagnostic split. Its result is reported separately,
not presented as source-group-held-out quality. Reciprocal complete groups
counterbalance the same two answers across target A and target B.

Our own portions use SAND. Source grants remain attached to unsuccessful
checkpoints as well as any successful delta. No protected pool material,
exposed development capture or evaluation-answer lookup is used.
