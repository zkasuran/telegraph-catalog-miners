# Source grants and scope

## ContractNLI [koreeda-manning-2021-contractnli-dataset]

The primary publisher states: "Our dataset is released under CC BY 4.0."
Download terms state that use is "in accordance with the terms and conditions of the Creative Commons Attribution 4.0 International Public License".
https://stanfordnlp.github.io/contract-nli/

The retained grant allows "reproduce and Share the Licensed Material, in whole or in part" and "produce, reproduce, and Share Adapted Material" under section 2(a)(1). Attribution, change notices and retained downstream rights apply. Source text, source annotations and constructed answer templates are distinct. Do not call our templates human-reviewed. No outside source links are fetched under the dataset's licence.

Pinned revision: eced6528dd3c1d14d73f9a87df8f7bdbc03126f9.
Archive SHA-256: e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b.
All 613 archive members, 607 originals, 48,058 sentence spans and 10,319 annotations are read and checked. Fifteen originals with restrictive wording are deferred. A keyword scan is not legal clearance. All selected source portions retain the publisher's CC BY 4.0 grant and attribution.

The NLI choices are document-scoped. Every annotated evidence span is retained, with no NotMentioned absence inference on excerpts. These are obligation/exception entailment labels, not performance compliance. Conditional applicability and definitions outside the selected evidence remain explicit limits.

## Reviewed runtime contract

Telegraph reference kit, Copyright (c) 2026 telegraphprotocol, MIT.
https://github.com/telegraphprotocol/telegraph-wasm-baseline/tree/1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd
The actual renderer uses `### Task:\n<prompt>\n\n### Answer:\n<answer>`. The batch executor echoes bare foundation, adapter and tokenizer hashes. Full engine execution has not occurred in this attempt.

The exact primary renderer is retained under MIT in reference_render.py. Its full source was checked at SHA-256 5da2919c66c2d0840104628ab88e98a9c8e5e578686f71ca9be4dcd91bd18d71 before extraction. Training/runtime use the entire pinned kit, not a substituted scoring implementation.

## ModernBERT [modernbert]

Answer.AI and LightOn's pinned primary card states: "We release the ModernBERT model architectures, model weights, training codebase under the Apache 2.0 license."
https://huggingface.co/answerdotai/ModernBERT-base/tree/8949b909ec900327062f0ebf497f51aef5e6f0c8
Apache 2.0 grants reproduction, derivative works and distribution subject to retained notices. Full terms are in licences.json and every package's upstream metadata. The unchanged backbone is fetched only on the public CPU runner. No local model loading or inference is allowed.

## Our additions

New code uses the reviewed LicenseRef-zkasuran-SAND-1.0 terms. Those terms do not restrict the upstream text or annotations. Current source-gate failure evidence retains the same licence boundary as a successful export. No trained model is exported yet.
