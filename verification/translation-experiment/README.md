# English and French translation preference experiment

Status: new reviewed source pool and fixed pilot pass source/data gates. A
guarded public CPU workflow is prepared. No model training, full package check
or reference-runtime scoring has run locally or remotely in this task.

## Primary source and exact grant

[Tatoeba Downloads](https://tatoeba.org/en/downloads) states:

> These files are released under CC BY 2.0 FR.

The [CC BY 2.0 FR legal code](https://creativecommons.org/licenses/by/2.0/fr/legalcode)
section 3(2) grants:

> créer et reproduire des Oeuvres dites Dérivées

Section 3(5) grants database extraction and reuse:

> lorsque l'Oeuvre est une base de données, extraire et réutiliser des parties substantielles de l'Oeuvre.

Section 4(b) requires attribution:

> Il doit communiquer le nom de l'Auteur Original ou son éventuel pseudonyme s'il est indiqué

It also requires the original title if supplied, source URI where possible and
identification of derived use. Section 4(a) requires the licence or licence URI,
retaining legal notices and not restricting the granted source rights. This is
CC BY, not share-alike or non-commercial. Full terms remain in `rights.json`.

The active pilot uses the **general CC BY 2.0 FR sentence exports**. It does not
relabel any source text CC0. `pilot-manifest.json` retains every source sentence,
ID, credited contributor/owner from the detailed export, source sentence URI,
profile reference, original/detailed source rows, timestamps and exact export
hashes. An exported owner is a credited contributor, not an independently
verified creation history. All selected sentences have that credit. The
sentence URI also exposes the publisher's sentence context and history.

Attribution: Tatoeba and contributors, https://tatoeba.org, CC BY 2.0 FR.
Every original remains under its applicable grant. Our SAND terms cover code
and additions only. No ordinary-source text is placed under blanket SAND.
The retained historical CC0 probe keeps its own CC0 source rights.

## Pinned original and link inventory

All files are from the November 14, 2026 weekly snapshot. Both plain and detailed
inventories match by sentence ID, language and unchanged text.

| Inventory | Observed count |
|---|---:|
| English originals | 2,041,097 |
| French originals | 730,700 |
| Originals without a credited owner, excluded from selection | 44,791 |
| Normalized duplicate originals | 2,148 |
| Full primary directed link rows | 28,651,608 |
| Reciprocal English/French links | 442,871 |
| Missing reciprocal link identities | 0 |
| English/French linked components after alias unions | 238,427 |

`general-sources.json` pins all four general/detailed files. The previously
checked link archive is reused at SHA-256
`08a802bfee9242fe2476836ec94e180588149f047451bce7519331b871b19e4f`.
All-language link connectivity is computed before source/donor selection.
Visible NFKC/case/whitespace aliases from both English and French are merged
across both languages. Only IDs are read for other languages. Original source
texts are never normalized, clipped, padded or corrected.

The historical CC0-only probe yielded 77 links across 68 components. That is
preserved as earlier evidence, not the active corpus count.

## Source review before the new freeze

The exposed failed freeze is retained byte for byte in
`archive/failed-ca65b547545b/`, with all five documented errors and 169 unresolved
pronoun/context flags. All 448 old components are excluded from the new pool.
The old final audit remains failed. No old source text or final label was fixed.

The implementation agent read 400 unsplit source/donor blocks, covering 800
actual source links. AI-assisted bilingual review approved 236 blocks and
rejected 164. Decisions and exact source/donor statements remain in
`source-pool-review.json` and `reviewed-source-pool.json`. These are AI reviews,
not human votes or professional certification. Unresolved number, referent,
gender, tense, unit, modal and word-sense cases were excluded before splitting.

Only then did a new deterministic hash rule select the fixed 128/32/64 groups.
The reviewed pool SHA-256 is
`b548e4d8fc297e1f66732db7830c84d826cf9158bca50275ec8442d1b37d6e6a`.
The new freeze SHA-256 is
`099fa6f2c0f4eda56dd995ed5265e820b8c885bae58667e0e72f12db1f7e6f7b`.

## Fixed constructed preference pilot

The intact, credited short-sentence scope provides 299,830 eligible original
links from 176,210 components. The conservative new first-person pool has
47,980 candidate links from 31,834 components after old-component exclusions.
Selection uses a fixed hash order, one linked
original per component and paired donor components with matched source/target
length buckets and punctuation. Content-word difference is a donor screen,
not semantic certification. The source grant does not certify link quality.

Each block contains two original English/French links from different complete
equivalence components. It makes four constructed preferences: two in each
direction, with target texts used reciprocally as chosen and rejected. Donors
stay in one split. This is a source-bound original translation versus unrelated
original donor experiment, not a minimal-error or universal quality benchmark.
The labels are programmatically constructed, not human preference votes.

| Split | Paired source/donor blocks | Distinct components | Preference rows |
|---|---:|---:|---:|
| Train | 128 | 256 | 512 |
| Validation | 32 | 64 | 128 |
| Frozen final audit | 64 | 128 | 256 |

There are 896 unchanged attributed sentence originals. Bilingual source and
donor review preceded splitting. All groups, identities and final texts were
frozen before numeric gate measurements.
There is no overlap by identity, component or visible normalized text between
splits. Both directions have equal counts. Length-direction controls are 50%
within each direction, with ties counted half. Target-role histograms match.

## Measured gates

The unchanged kit at revision `1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd`
passes its actual data gate on train, validation and final rows. Unique rejected
fractions are 100%, length shortcuts 50% and held-out negative overlap zero.
Train-fitted lexical overlap scores 53.03%, 51.56% and 51.56%, below the unchanged
65% bar. The released tokenizer matches all 88 conformance vectors.

All 5,376 rendered source/answer texts across raw, JSON and prose wrappers fit
256 tokens, with a maximum of 72. The full chosen/rejected distinction remains
visible. All request bodies fit 256 UTF-8 bytes.

The kit's random row holdout splits reciprocal answer roles and produces
unusually low surface-probe scores. Those numbers are retained verbatim in
`pilot-checks.json`. They are not independent semantic evidence. Within a whole
reciprocal group, an answer-only ranking wins one preference and loses its
reverse, so its pair accuracy is 50%, with ties counted half.

## Archived failure from the previous exposed freeze

The implementation agent read all 448 original links and 224 donor blocks.
This was AI-assisted bilingual review, not human or professional certification.
Five documented reference failures affect ten constructed rows:

| Original IDs | Failure |
|---|---|
| 9795147 / 2564823 | Present progressive changes to imperfect past. |
| 11008290 / 6212792 | Reporting inability changes to reporting why one is unable. |
| 10092686 / 10971287 | The requirement to book in advance disappears. |
| 12997332 / 5251372 | Plural-possessor ownership changes to singular-possessor ownership. |
| 845613 / 844653 | Inability to act accordingly changes to inability to agree. |

The archived `semantic-review.json` binds exact original excerpts and split
locations to the old frozen manifest hash. This rejected the old pilot although numeric
gates pass. No source correction, replacement or reselection follows the final
audit. A direct reciprocal link is provenance, not proof of semantic fidelity.

The new pool excludes these entire old components. Neither these errors nor
the original failed final audit are relabelled as valid. The active source review
is bound to a different, newly sealed pilot and remains disclosed as AI-assisted.

## Guarded public CPU experiment

`.github/workflows/evaluator-translation-train.yml` is manual-dispatch only,
public-repository only and uses one standard `ubuntu-24.04` CPU runner. It has
read-only repository permissions, pinned actions, hash-locked CPU dependencies,
four threads, a 330-minute job cap and a 16,800-second process-tree wall cap.
The supervisor stops above the lower of 10 GiB RSS and 75% of measured runner
memory. These are ceilings, not measured completion or memory promises.

`remote_train.py` verifies primary original text, attribution and reciprocal link
bytes again, then reruns unchanged kit data gates and source-class preflight
before model loading. It reuses the pinned shared foundation preparation helper,
ModernBERT LoRA/model/export functions and controller parser/runtime contracts.
Controller files are retained unchanged. Only the generated parser build copy
specializes the family constant to LANGUAGE_TRANSLATION.

The recipe is fixed: rank 8, alpha 32, 256 tokens, one epoch, batch one,
accumulation 16 and 32 optimizer steps. Final scores do not select the recipe.
Require 80% ordering in every raw/JSON/prose format, 70% in both directions and
claimed length subsets, five points over a seed-matched untrained adapter and
ten points over the strongest frozen shortcut. No threshold is lowered.

Successful native quality yields an honestly labelled source-bound candidate,
with parser/runtime still pending. Full unmodified Stage 1 checks then validate
every header/tensor requirement. The reference executor checks all formats,
repeated bits, a fresh process, wrong identities and native/reference numerical
and ordering agreement. Node qualification remains separate and is not attempted.

## Reproduction and checkpoint rights

Set `SOURCE_OUT` to an existing validated corpus cache. It must contain
`general.sqlite`, `general-index.json`, `general-graph.json`, the four pinned
general/detailed sentence archives and `links.tar.bz2`. The audit checks the
cache reports against frozen source pins, reads SQLite in read-only mode and
verifies original text, attribution, component membership and archive hashes.
Missing or mismatched caches are refused without creating a database.

For a fresh cache, choose a writable empty `SOURCE_OUT` directory and run:

```sh
python3 verification/translation-experiment/general_sources.py capture --out "$SOURCE_OUT"
python3 verification/translation-experiment/general_sources.py index --out "$SOURCE_OUT"
python3 verification/translation-experiment/general_sources.py graph --out "$SOURCE_OUT"
```

Capture obtains the publisher description and pinned archives, including links.
It never rewrites an existing frozen source specification. Mutable download URLs
must still return the exact recorded snapshot hashes; otherwise obtain those
matching archives separately. Index and graph commands refuse to overwrite
existing source state. Verification never selects a new source pool or freeze.

From the checkout, with the existing source-only Python environment and pinned
kit/tokenizer paths, put controller Ruff 0.16.4 on `PATH` using its default
configuration, then run:

```sh
SOURCE_PYTHON="$SOURCE_PYTHON" KIT="$KIT" TOKENIZER="$TOKENIZER" SOURCE_OUT="$SOURCE_OUT" bash verification/translation-experiment/verify.sh
```

This runs parser and metadata-retention tests, Ruff, primary source hash/identity
checks and all real data/tokenizer gates. Derived JSONL evidence is written to
this checkout's ignored `out/pilot/`, not to the input corpus cache. It has no
patch-generation or private-packet dependency. It passes only when the reviewed source/data checks clear. It never
imports neural tooling or loads a model. The old failure is tested as an immutable
archive, not rerun or silently repaired.

`checkpoint_terms.py` uses the unchanged container-stamping helper with this
family's full source rights and complete per-sentence attribution manifest.
A synthetic empty-container test proves those fields survive stamping without
changing payload bytes. The complete attribution manifest uses a hash-checked,
bounded zlib-base64 encoding to fit the unchanged one-MiB header ceiling; no
source or attribution field is removed. Full source terms remain readable and
the plain manifest accompanies artifacts. Missing author references or unreviewed
share-alike grants are refused. Initial and trained checkpoints, reward-head ZIP
files and successful packages retain the same source/kit/foundation rights.
Only stamped files are eligible for the workflow's artifact upload.

No credentials, protected pool rows, local neural work, shared helper edits,
commits, pushes, service changes or workflow dispatches were performed.
