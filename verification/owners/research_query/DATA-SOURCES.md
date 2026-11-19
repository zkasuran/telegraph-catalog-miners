# Source grants and lineage

## ELIFE-XML

Primary publisher repository:
https://github.com/elifesciences/elife-article-xml/tree/c3cd502f94edfb5aa0c5870ed9eff053773c6125

The publisher README says all articles are available under CC BY or CC0.
The implementation does not infer a blanket grant from that statement. Each
selected original must have exactly one recognized licence URI, author names
and its own DOI. It retains the original licence statement, copyright notice,
title, source, version and modifications. XML bytes are checked against both
SHA-256 and the immutable Git blob identity. Only ordinary research article
abstract paragraphs without cited, formula or graphic elements are eligible.

## CC-BY-3

https://creativecommons.org/licenses/by/3.0/legalcode.txt

Section 3(a): "to Reproduce the Work". Section 3(b): "to create and Reproduce
Adaptations". Section 4 requires the licence URI and original author/title/source
credit. Changes must be identified. Upstream grant remains unchanged.

## CC-BY-4

https://creativecommons.org/licenses/by/4.0/legalcode.txt

Section 2(a)(1)(a): "reproduce and Share the Licensed Material, in whole or
in part". Section 2(a)(1)(b) also permits sharing adapted material.
Section 3 requires attribution and modification notices. Section 2(a)(5)(b)
prohibits restricting the original licensed material through downstream terms.

## CC0

https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt

Sections 2 and 3 permit exercise "for any purpose whatsoever". Section 4 does
not clear third-party rights. Names, logos, figures and external quoted material
are not included in the selected text.

## KIT-TRAIN

https://github.com/telegraphprotocol/telegraph-wasm-baseline/tree/1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd

MIT grant: "Permission is hereby granted, free of charge, to any person
obtaining a copy". The copyright and permission notice must remain included.
The full primary parser is retained. Its live-class hook is used only in an
isolated checker process, never to change a controller catalog.

## FOUNDATION-RELEASE

https://github.com/telegraphprotocol/telegraph-foundation

The primary README states that the released weights and tokenizer are the
unchanged ModernBERT-base assets under Apache-2.0. It binds the engine to the
kit's float_runtime/engine build, using oxionnx 0.1.7. The release repository
has no LICENSE file. The engine therefore retains the primary kit and oxionnx
source grants rather than inheriting a separate blanket release grant. The
controller must keep those engine notices when reproducing runtime evidence.

## MODERNBERT

https://huggingface.co/answerdotai/ModernBERT-base/tree/8949b909ec900327062f0ebf497f51aef5e6f0c8

The pinned model card declares Apache-2.0. The grant permits "reproduce,
prepare Derivative Works of". Full Apache terms and source notices must remain
attached. No base weights are downloaded or loaded by the local source gate.

## Label and licence boundary

Original text is distinct from questions, opposite-direction statements and
preferences. Those additions are constructed with AI assistance, not human
reviewed. They label agreement with the retained sentence, not global truth.
QASPER originals were not used because the provider annotation grant did not
settle rights in the older underlying paper text. No protected evaluation-pool
rows or miner captures enter this dataset.

Our additions use the reviewed SAND terms. That does not claw back the original
CC grants. Original excerpts and modified statements retain their source licence
and attribution. The same boundary applies to failed checkpoints.
