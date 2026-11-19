<!-- SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0 -->
# Bounded QA preference experiment

Status: **SOURCE GATES PASS, REMOTE MODEL/RUNTIME UNEXECUTED**. The complete
guarded CPU experiment is implemented. No model was loaded, trained or scored
locally. This is not a model quality or node-qualification result.

The source task is English single-turn answer relevance and usefulness under
OASST1 human sibling preferences. These ranks do not certify factual correctness.
They do not establish node input, public ordering or hidden-pool equivalence.

## Exact sources

`sources.json` pins every fetched file by revision and SHA-256. The OASST1 data
revision is `fdf72ae0827c1cda404aff25b6603abec9e3399b`. Its
[full Apache terms](https://huggingface.co/datasets/OpenAssistant/oasst1/raw/fdf72ae0827c1cda404aff25b6603abec9e3399b/LICENSE)
are included unchanged in `UPSTREAM-APACHE-2.0.txt`. Section 2 grants a "copyright license to reproduce,
prepare Derivative Works of". Section 4 requires full terms and change notices.
`NOTICE` names contributors, source grants and modifications.

Our own code and trained additions use `LicenseRef-zkasuran-SAND-1.0`. `LICENSE`
is the byte-exact reviewed catalog SAND text. OASST1 text and original human
annotations retain Apache-2.0. ModernBERT portions retain Apache-2.0 and Telegraph
portions retain MIT. Those grants are not narrowed by our own terms. The frozen
source manifest, source rows and original annotations remain byte-exact.

The [original reward data code](https://github.com/LAION-AI/Open-Assistant/blob/f1e6ed9526f5817531f3ab85441a40b3671ddccb/model/model_training/custom_datasets/oasst_dataset.py#L138-L151)
sorts sibling rank ascending and splits whole trees. Smaller rank is preferred.
Only original ranked root replies are paired. No negative text is fabricated.

The pinned Telegraph kit is `1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd`.
The base/foundation preparation function in `verification/training/run.py` is
executed unchanged after all data gates pass. This uses ModernBERT-base revision
`8949b909ec900327062f0ebf497f51aef5e6f0c8` and the released
`modernbert-base-lora16-v1` foundation, engine and tokenizer hashes. Existing
hash-locked CPU dependencies are reused. No old trained adapter is used.

## Source filters and boundaries

Both prompts and answers need English text, explicit `synthetic=false`, explicit
`deleted=false`, positive review and one explicit zero PII label. PII=0 is a source
annotation, not an independent privacy audit. Original semantic review and rights
exceptions remain unverified.

All 88,838 source messages retain their whole source-tree identities. Trees
connect only through NFKC, casefolded and whitespace-normalized compact root
questions or candidate answers that can reach the trainer. Unused descendants
cannot merge unrelated root tasks. Every connected component crossing official
train/validation is removed. One original preference is selected by pair identity
hash per remaining component. Official validation is the final audit.
Development splits only official train by a fixed group hash.

The fixed counts are 128 train, 32 development and 64 final audit. Each needs
equal original longer-preferred and shorter-preferred pairs. Audit selection is
frozen first. Failure never reduces the training gate or selects another recipe.
Smaller preserved files are labelled prototypes and cannot enable training.
Both complete rendered answers must fit the released tokenizer's 256-token cap
and retain different token sequences. Nothing is cropped or relabelled.

## Measured source-only result

The corrected trainer-input grouping removed three cross-official components.
The audit pool has 69 longer-preferred and 34 shorter-preferred components.
It supplies all 64 required audit pairs without changing any threshold or
source preference. OASST1 alone clears the fixed counts.

| Fixed split | Original pairs | Independent groups | Unique rejected fraction |
|---|---:|---:|---:|
| Train | 128 | 128 | 100% |
| Development | 32 | 32 | 100% |
| Final audit | 64 | 64 | 100% |

All 224 pairs fit intact. All 88 released tokenizer conformance
vectors match. No normalized selected text or group crosses splits.
`evidence/data/data-report.json` binds counts and split hashes.

The unchanged kit training dataset gate passed: length shortcut 50%, reused
negative share 0.78%, cross-split negative leak 0% and its five-shuffle
answer-only surface probe 46.84%. Train-fitted development controls are recorded
in `evidence/data/dataset-gate.json`. Final audit control accuracies stay sealed
until a fixed training checkpoint exists. No model result is claimed.
Development controls: character length 50%, actual token length 50%, TF-IDF
overlap 46.875%, answer-only 62.5% and surface style 50%. Every control is
below the unchanged 65% ceiling. `contract.json` fixes counts, exact split
hashes and quality thresholds before any model phase.

The original train-only controls are bundled byte-exact in
`baselines/frozen-controls.json`, SHA-256
`4aa549339b120184258d6d3c17b5f407f6aa1562c2973e7e4b65bfee6ea8491b`.
Its manifest binds 128 training originals, their exact source split hash,
source revision and released tokenizer. Validation checks dimensions, finite
coefficients, vocabulary indices and vocabulary/IDF correspondence to that
training source. Production scoring replays immutable copied coefficients with
scalar `math.fsum` reductions. It never refits on development or audit.

The reported run 44463468482 failed while requiring byte-identical NumPy/BLAS
refitting. That failed refit is not relabelled as passed. Fitting the same recipe
again is not a portable byte identity guarantee. An optional `--diagnose-refit`
reports fresh train-only coefficient drift separately; it never replaces the
baseline or changes acceptance thresholds. Audit control results remain sealed
until the development-selected neural checkpoint exists.

## Entry points

Source-only prep can run locally using an exact verified source cache:

```sh
python verification/qa-experiment/data.py --out verification/qa-experiment/evidence/data --source-cache /path/to/verified-public-sources
python verification/qa-experiment/gates.py --data verification/qa-experiment/evidence/data --kit /path/to/pinned-kit
bash verification/qa-experiment/verify.sh
```

Controller remote entry point, within `.github/workflows/evaluator-qa-train.yml`:

```sh
timeout --signal=TERM --kill-after=20s 18000 "$RUNNER_TEMP/qa-venv/bin/python" verification/qa-experiment/run.py --root "$RUNNER_TEMP/qa-experiment"
```

The guard requires the assigned public GitHub repository and runner temporary
storage. Total workflow ceiling is 330 minutes. The experiment alarm is 300
minutes including downloads and source prep. Fixed CPU settings: four threads,
fp32, rank 8, alpha 32, batch 1, accumulation 16, evaluation batch 1 and three
epochs. That is 24 optimizer steps for 128 pairs. All three epochs run, with no
early stopping or recipe search. Each optimizer step saves a licensed recovery
checkpoint with measured throughput. Projected training above the reserve stops.

The best epoch has the lowest development loss. Exact ties keep the earlier
epoch. Development evaluation runs inside the primary CPU Torch `fork_rng`
context and restores Python/NumPy state and model mode. The training sampler has
its own seeded generator. Actual state equality is checked around every remote
development evaluation. The selected trainable state is restored and its hash
verified before the sealed audit is scored. No audit metric selects an epoch.

After a fixed checkpoint, source-held-out trained ranking is compared with the
seed-matched untrained head and LoRA state. Gates require 80% aggregate ordering,
70% in each claimed length subset, five points above untrained and ten above the
strongest frozen shortcut. Group Wilson intervals accompany the audit result.
These are experiment gates, not organizer rules. Audit never selects a checkpoint.

## Full live-class parser and runtime integration

`live_class.py` fetches only the exact anonymous family resolution and child
schema HTTPS endpoints. The snapshot expires after 15 minutes. The Go wrapper
validates URLs, timestamps and matching family/child classes before calling
upstream `catalog.UseClasses` and the unchanged `adapterbuild.CheckLoRA`.
It preserves every header, class, hash, rank, tensor shape, dtype, offset and
finite-value check. The training record names the exact QA family and child.
The null QA response schema does not acquire a fabricated sentiment schema.

Source-only tests exercised the full upstream parser and `CheckLoRA` using tiny
artificial byte fixtures. No fixture was loaded or scored as a model. The live
QA preflight was also executed with the compiled wrapper and reports judgment.

After the development-selected checkpoint, the remote path exports only rank-8 LoRA deltas and
the trained reward head. It records native scores for all 64 audit pairs, runs
the complete Stage 1 package parser and starts the pinned reference executor.
All 128 token vectors are scored twice with identical engine bits. Eight vectors
are checked in a fresh process. Wrong foundation, tokenizer and adapter request
identities must be rejected by the verifier. The executor reports actual identity
rather than rejecting such requests itself, as its primary code specifies.

Native-versus-WASM scores must agree within fixed absolute/relative tolerance
0.001 and preserve every audit pair's strict ordering. This is numerical and
ordering comparison, not an exact-bit claim across different runtimes. All model
quality thresholds still apply. Passing creates a source candidate manifest for
the controller's separate node test. The code never registers, signs, dispatches
or spends. Model training, real exported-package validation and real runtime
scores remain unexecuted here. Their results cannot be called PASS in advance.

Every exported delta embeds the full own SAND text plus separate unchanged
Apache/MIT terms and upstream attribution. Initial, selected and interrupted
PEFT checkpoints receive the same byte metadata without changing tensor data or
descriptors. Reward-head `.pt` files receive an inert licence record inside the
existing PyTorch ZIP prefix. Original storage members and offsets stay unchanged.
The actual `weights_only=True` reward-head roundtrip is checked remotely before
audit scoring. Local tests checked the container transformation only.

Checkpoint folders also carry full separate licence sidecars. Upload is allowed
only after the final preservation gate succeeds. The source text and annotations
are never relicensed as our own. Frozen diagnostic coefficients receive separate
own-addition terms without changing their freeze hash or restricting source
tokens. Raw cached sources and source-derived rows are locally ignored, while
hash/count evidence remains in the patch.
