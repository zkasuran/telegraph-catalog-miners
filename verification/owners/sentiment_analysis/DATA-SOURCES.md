# Data and model source grants

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
was exposed. This continuation changes only ignored registration metadata.
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
