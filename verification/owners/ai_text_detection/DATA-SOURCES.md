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
