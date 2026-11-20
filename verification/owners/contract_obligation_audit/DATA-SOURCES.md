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

New code uses the reviewed LicenseRef-zkasuran-SAND-1.0 terms. Those terms do not restrict the upstream text or annotations. Failed checkpoint/source-gate evidence retains the same licence boundary as a successful export. An exported failed model is not a qualified candidate.

The first owned model was trained in public CPU run 44564516958 and retains all full upstream grants in its package. It failed quality and is not registered. Its newly exposed 56 audit groups join the earlier 43-group retirement. Neither group set may enter later training, development or independent audit.

MAUD is being inspected as an alternative source under the publisher's CC BY 4.0 grant. The complete 158-member archive and all CSV rows were read. Main contexts, abridged contexts and rare-answer constructions are distinct. Annotation-added page locators are metadata, not contract wording. No MAUD preference is called freshly human-reviewed. No MAUD model or audit score has been produced yet.

## Generation-three raw judgment context

The primary training standard specifies raw prompt/answer strings and up to 512 tokens. Evaluation-material documentation permits raw prompt lengths up to 16000 characters. These are scoring-string contracts, not the typed miner's separate 256-byte slot.
https://raw.githubusercontent.com/telegraphprotocol/telegraph-wasm-baseline/1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd/float_runtime/train_adapter.py
The pinned reference engine already executed a 512-token fixed vector in the first actual run. Generation three retains complete original annotation evidence up to 768 bytes and filters actual rendered strings through the pinned tokenizer. All 980 permitted original records fit within 426 tokens. No source words, conditions or annotated spans are dropped. Typed long-contract miner delivery and URL dereference remain unverified.

The source selection continues the same four claims and excludes all 99 retired groups plus 227 prior exposed groups from its fresh audit. Its semantic cue coefficients are fitted only on training and frozen as integers. Their measured scores strengthen the model-quality comparison. No audit label or score selects the new candidate.

Generation four keeps the exact generation-three source audit IDs/groups because no independent audit executed in run 44612876956. Its broader training/development sources are excluded from those groups. Direct statement and explicit logical-complement candidate forms come from the published Entailment/Contradiction choice, never a made-up performance observation. No NotMentioned premise is treated as a contradiction. The source labels remain original, while the response construction is AI-assisted and disclosed.

## Generation-five broader training

Run 44628637805 failed native validation in both formats (0.6333 overall) and never executed its independent audit. The same 64 audit sources in 61 groups remain fixed outside all training/development. All 99 earlier retirements remain excluded. Training now uses 320 original pairs in 182 source groups; validation remains the identical 30 original pairs in 25 groups. No audit score informed this selection.

The actual train loss fell to 0.0001 while validation loss rose to 2.9184. The recovery continues the exact latest licensed checkpoint with six epochs, learning rate 0.00005 and weight decay 0.05. The primary pinned trainer saves the classifier in PEFT modules_to_save as well as reward_head.pt; the retained header was checked without loading tensors. Both formats, unchanged worst-format validation selection and every 80/70/+5/+10 gate remain mandatory. Source baseline coefficients are refitted only on training and frozen as integers. Source publication is not a neural-quality result.

## Generation-six auxiliary training

Run 44647112954 improved native validation to 0.7333 in both formats but still failed the unchanged quality gate. Its audit never executed. The retained best licensed checkpoint is resumed, not registered or retried unchanged. The identical 320 primary training pairs now have 36 additional balanced original training records across five auxiliary hypotheses (nda-10, nda-19, nda-2, nda-4 and nda-5). All published source labels, complete spans and CC BY 4.0 attribution remain attached. Auxiliary claims are training coverage, not newly audited capability claims. Original development and audit bytes remain exactly unchanged and all 99 permanent retirements remain excluded.

The complete remaining training inventory was counted before selection. A proposed 40 auxiliary rows yielded only 36 under the existing normalized-evidence and four-per-group cap. No positive-only or NotMentioned annotation was made into a fabricated contradiction. The actual pinned tokenizer checks every selected training/development view without truncation. Frozen coefficients are fitted on training only.

The primary pinned trainer's Bradley-Terry loss is `-logsigmoid(chosen_reward - rejected_reward - margin).mean()`. Our training addition penalizes squared reward differences with coefficient 0.01 to limit overconfident margins. This regularizer is removed before native validation measurement and never changes reference-engine scoring. Four epochs, learning rate 0.000025 and validation-only worst-format accuracy selection remain bounded. A learned improvement is unverified until the real remote results arrive.

## Generation-seven retained rank growth

Actual run 44678726514 failed native validation at 0.6667 JSON and 0.7333 prose. Its audit was skipped. All source split bytes are unchanged in generation seven. Validation alone selected the earlier licensed source-trained checkpoint from run 44647112954 because its worse-format accuracy was higher (0.7333). No independent audit score selected either checkpoint.

The pinned primary `lora_format.py` fixes `R_MAX = 16` and one scale `alpha/rank`. The primary exporter supports F32 factors with rank up to sixteen, all four target linears per layer and the unchanged reward head. Our retained PEFT header contains all 176 rank-eight factors. Expansion preserves those values, grows rank to sixteen, sets alpha to 64 to retain scale four and initializes the new output factors at zero. New input rows use seeded bounded initialization. No backbone merge, binary wrapping or new external model is used. These tensor operations run only in the bounded public CPU workflow. Actual preservation checks emit rank-growth.json; they do not establish reference-engine bit identity or quality.

Eight epochs, learning rate 0.00005 and reward-margin penalty 0.001 train the expanded capacity. Full native validation, current parser, pinned reference-engine execution and the identical unconsumed source audit remain mandatory. Every quality and identity threshold is unchanged.
