# General response-quality training source

[H1] NVIDIA HelpSteer2, Wang et al. 2024, arXiv:2410.01257.
Primary card at pinned revision:
https://huggingface.co/datasets/nvidia/HelpSteer2/raw/990b2711a36180dd19d9c94b8627844866f8982a/README.md
It declares `license: cc-by-4.0` and recommends the dedicated comparative
annotations for Bradley-Terry modelling. Negative strengths prefer response 1.
Positive strengths prefer response 2. Zero and -100 are not ranking labels.
The retained primary card and full Creative Commons terms accompany private
prepared data. The source grant is not a claim of new human review.

[H2] Creative Commons Attribution 4.0 section 2(a)(1) grants permission to:
"reproduce and Share the Licensed Material, in whole or in part" and
"produce, reproduce, and Share Adapted Material."
https://creativecommons.org/licenses/by/4.0/legalcode.txt
Source attribution, licence link and notice of our transformations remain
attached. SAND covers our code only, not extra restrictions on NVIDIA data.

Changes: source sign normalized, invalid/neutral/disagreeing annotations excluded,
normalized prompt/answer components grouped and one original selected per group.
No response text rewritten. Rating-derived copies are not additional examples.
Only the original train partition is eligible; val/validation rows are skipped
before response inspection. Separate official evaluation files remain unopened.

This is general preference development for the text-judgment component.
It is not verified VAT truth, a task-specific independent audit or proof that
the live Vendor slot can be won. Entity/time/source-domain independence remains
unverified. Exact tokenizer retention and actual neural quality run remotely.

The first development job passed the dataset gate but failed its combined token
retention/conformance check before training. A byte cap does not prove token
retention. The next revision reports every train/validation text's token-check
outcome as aggregate counts before refusing any mismatch or truncation. It does
not alter source splits, inspect the sealed audit or weaken the 512-token gate.

[M1] ModernBERT by Answer.AI and LightOn, revision
8949b909ec900327062f0ebf497f51aef5e6f0c8. Primary model card:
https://huggingface.co/answerdotai/ModernBERT-base/raw/8949b909ec900327062f0ebf497f51aef5e6f0c8/README.md
Its explicit grant says: "We release the ModernBERT model architectures, model
weights, training codebase under the Apache 2.0 license."
The full Apache terms remain in licences.json with original notices. The remote
build fetches only pinned model/config hashes; no foundation/model loads locally.

[K1] Telegraph reference kit, Copyright (c) 2026 telegraphprotocol, MIT,
revision 1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd. Its full grant and copyright
remain in licences.json. Permission includes "to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software" subject to
retaining its notice. Actual primary training functions are byte-checked before
use. Native validation metrics are not the node reference-engine results.

The existing hash-pinned CPU dependency lock is retained, not freshly resolved.
Remote dependency audit must check every installed release without skips.
The original train-fitted BASELINES.json bytes are loaded on the runner rather
than requiring a fresh floating-point fit to reproduce the source split proof.

The completed diagnostic checked all 384 development texts. One training text
exceeded the cap at 583 tokens. Validation fit within 455 tokens and every capped
training/node vector agreed. Recovery excludes only whole over-cap training
pairs under the exact pinned tokenizer. It preserves the original split bytes,
validation and sealed audit. No source text or annotation is changed.
The unchanged dataset and token gates run again on retained training before
neural fitting. Revised training-only coefficients are frozen once and retained
in an attributed artifact. Both original and revised frozen shortcuts must be
beaten. Capacity exclusions cannot hide identity errors or validation truncation.
# Frozen-feature development boundary

The first remote LoRA experiment scored 56.25% on validation and lost to the
60.9375% length shortcut. The replacement fits only the supported linear reward
head on frozen primary classifier-input features. Four predeclared ridge values
are fitted on training only and selected on unchanged validation. All source
annotations, original split bytes and both sealed audits remain unchanged.
Answer-only, rotated-prompt and prompt-only controls measure attribution. They
are not new source annotations or an independent truth set. No reference-engine
qualification is claimed by native CPU scoring. Source text and fitted artifacts
retain NVIDIA HelpSteer2 attribution and CC BY 4.0. Competitive additions use SAND.

## Expanded adaptation development

The completed frozen head fit all 127 training pairs but scored 59.375%
on unchanged validation, below the 60.9375% shortcut. The next experiment
keeps all original split allocations and adds 512 unused source groups only
to training. A frozen-head control and two last-four-layer LoRA epochs are
compared on the same validation. The earliest stage wins a tie. New classical
coefficients fit once remotely on retained training and their exact bytes stay
in the result artifact. Both earlier frozen controls still count. No source
text or annotation is rewritten. No audit is used for selection. Native scores
do not prove alignment with the published foundation or reference engine.
No exact Vendor audit or registration-ready result is claimed.
