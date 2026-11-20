# Source grants and training boundary

SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0

## Civil Comments and Jigsaw annotations

Primary definition: https://www.tensorflow.org/datasets/catalog/civil_comments

Grant read on November 20, 2026: "This dataset is released under CC0, as is the
underlying comment text."

The primary definition says labels are the fraction of annotators assigning the
attribute. This experiment preserves toxicity exactly zero as Safe and at least
0.8 as Flagged. Original public comments and source annotations are distinct from
constructed preferences and learned predictions. No human review of generated
preferences is claimed.

Pinned files and SHA-256 values are retained in support/training/sources.json.
The original 768 train and 128 validation originals are reproduced byte-for-byte.
Training originally used a bounded pool from the first training shard, not a
population sample of the full corpus. The independent 256-original audit comes
from a complete pinned official test-file scan. Original train/validation/test
groups and conflicting groups are excluded before fixed-salt selection.

Source inventory: the official default corpus contains train, validation and test
splits. TFDS documents 1,804,874 train examples and 97,320 in each remaining split.
This is inventory context, not a claim that this experiment trains on all rows.
No parent text or identity annotations enter the experiment. Parent-context,
identity subgroups, middle-range toxicity and multilingual policy are unvalidated.

F3 warm-start assets are the licensed trained H9 epoch 8 checkpoint and the
separate genuine untrained initialization from public run 44678239020. Their
entire archive and individual SHA-256 values were checked without model loading.
Both tensor containers retain original grants in their metadata. Frozen
train-fitted baseline coefficients from that same job are reused byte-for-byte.
These assets are fitted training work, not independent audit results or human
review. H9 never opened the audit. The original failure evidence remains retained.

The audit stays sealed through validation selection. Its hash and exclusion hash
are frozen in experiment.json. Exposure cannot become training data or select a
later candidate. Three response views are correlated views of the same originals.

## Foundation and reference source

ModernBERT: https://huggingface.co/answerdotai/ModernBERT-base/tree/8949b909ec900327062f0ebf497f51aef5e6f0c8

Apache-2.0 section 2 grants a "perpetual, worldwide, non-exclusive, no-charge,
royalty-free, irrevocable copyright license to reproduce, prepare Derivative
Works of, publicly display, publicly perform, sublicense, and distribute the
Work and such Derivative Works in Source or Object form." Full terms remain in
the source bundle and exported bytes. The original grant governs those portions.

Telegraph kit: https://github.com/telegraphprotocol/telegraph-wasm-baseline/tree/1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd

MIT permits dealing "in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software" subject to preserving the notice. Original
MIT copyright and full terms travel in the bundle and checkpoint metadata.

Our competition additions use the reviewed SAND terms in LICENSE. Support source
is fingerprinted in SUPPORT.json. No upstream compiled adapter is wrapped or
re-registered. No protected pool, provider account or signer credential is used.

## Qualification recovery

Public GitHub artifact 14621277811 from F3 run 44703787106 is an immutable
development/model evidence input, not new training data. Its whole archive and
23 required members are SHA-256-bound in recovery.json. Exact package bytes keep
the original Apache-2.0, MIT and SAND grants above. Civil Comments remains CC0.
The audit is the same 256 groups selected before training, not a fresh draw.
Completed trained reference streams are retained. Recovery performs no fitting.

## B3 source-anchored training

Recovery completed with a genuine bare-label Flagged audit loss. That entire audit
is now an exclusion set, not new training or validation data. B3 uses only the
original 768 training groups. Each group rotates through JSON, prose and a
label-led response quoting the actual source text. These are constructed
preferences under the retained source grants, not human-reviewed preferences.
Exact bare labels remain unseen transfer inputs. No padding or random tags are
added to evade the upstream repeated-negative gate.

The new 256-group audit is frozen before model loading from a full pinned official
test-file scan with a fixed new salt. All original split groups and all 256 exposed
audit groups are excluded. Source conflict removal and normalized identity checks
remain mandatory. New train-only fitted baseline coefficients are emitted once
then consumed as immutable bytes for validation, audit and reproduction.

Run 44745877586 failed the train-only shortcut source gate during preparation.
No foundation model was loaded and no validation or audit model score was made.
The corrected constructed responses identify the accepted verdict and explicitly
exclude its opposite, quoting the same original source in both answers. Role
reversal changes truth while keeping word bags and lengths equal. This removes
surface tells without lowering the original gates or adding meaningless padding.
The actual still-unscored audit hash and all group identities are now pinned.
