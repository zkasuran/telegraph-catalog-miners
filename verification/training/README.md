# Empty-family training experiments

This workflow trains a new ModernBERT LoRA reward adapter from published source
labels. It checks disjoint source groups, shortcut baselines and the exported
Telegraph runtime. It runs once on a standard public CPU runner and stops within
330 minutes. It is a development experiment, not continuous serving.

## Scope

| Family | Training source | Held-out boundary | Unchecked coverage |
|---|---|---|---|
| SENTIMENT_ANALYSIS | UCI Sentiment Labelled Sentences | Yelp originals never reach training or validation | Neutral, Mixed and non-English text |
| CONTENT_MODERATION | Civil Comments, pinned first train shard | Official validation and test files, normalized duplicates removed | Ambiguous toxicity and complete moderation policy |

Each split contains exactly balanced supported labels. We strip no meaningful
source text and accept only originals between 8 and 256 UTF-8 bytes. The fixed
subsets contain 768 training, 128 validation and 256 test originals. This is not a
population benchmark. Civil Comments uses toxicity 0 for Safe and at least 0.8
for Flagged; that proxy is narrower than Telegraph's policy categories.

The source labels are published annotations. The preference responses and wrong
label substitutions are constructed by AI-assisted code, not human preference
annotations. These files are not uploaded to the evaluation pool.

## Gates

- Hash-check every source, base weight and foundation input before use.
- Remove normalized cross-split duplicates and contradictory source labels.
- Run the pinned kit's dataset gate without a bypass.
- Use validation only for best-epoch selection.
- Compare the trained adapter with its seed-matched untrained initialization on
  the same test originals in JSON, prose and bare-label response formats.
- Require at least 80% pair accuracy in every format, at least 70% per supported
  label and five points of improvement over the untrained adapter.
- Check the real exported foundation runtime on the held-out pairs and repeated
  identical token vectors. Node qualification remains a separate test.

Length and answer-only surface probes are part of the upstream dataset gate.
Additional length, TF-IDF overlap and alarm-keyword rules fit their direction on
training only. Every split must remain below 65% for those frozen shortcuts.
Source labels and a normalized-group manifest remain distinct from measured
model quality. Test variants are repeated views of 256 originals, not extra
independent examples.

## Run

```sh
bash verification/verify.sh
gh workflow run evaluator-train.yml -f family=SENTIMENT_ANALYSIS
gh run download <run-id> -n training-SENTIMENT_ANALYSIS -D training-evidence
```

Training and inference refuse to run outside this repository's GitHub workflow.
The workflow has read-only repository access and receives no provider or wallet
credentials. It does not register a candidate or create an external endpoint.
Generated data is not included in the public artifacts.
The best saved epoch is retained even when the quality gate fails. Its tensor
container carries the terms; this checkpoint is not a qualified node package.

## Sources and terms

Source revisions and SHA-256 values are fixed in `sources.json`. Full upstream
MIT and Apache-2.0 terms accompany our SAND terms in `licences.json` and are
embedded in every exported delta package. Third-party rights remain unchanged.
Our terms apply only to our own code and trained additions.

UCI: Dimitrios Kotzias (2015), Sentiment Labelled Sentences,
https://doi.org/10.24432/C57604, CC BY 4.0. The repository grants sharing and
adaptation for any purpose with appropriate credit. Source text and labels are
retained; filtering and preference-pair construction are our changes.

Civil Comments: https://www.tensorflow.org/datasets/catalog/civil_comments.
The primary catalog states: "This dataset is released under CC0, as is the
underlying comment text." Parent text is not used. Borkan et al. (2019),
https://arxiv.org/abs/1903.04561, describes the toxicity annotations.

ModernBERT: Answer.AI and LightOn, Apache-2.0. The model card states: "We release
the ModernBERT model architectures, model weights, training codebase under the
Apache 2.0 license." The pretrained backbone is unchanged. The trained reward
head and low-rank deltas are new.

Telegraph reference kit: Copyright (c) 2026 telegraphprotocol, MIT. The grant
includes the rights to "use, copy, modify, merge, publish, distribute, sublicense"
subject to retaining its copyright and permission notice.
