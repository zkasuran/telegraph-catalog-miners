# Moderation schedule diagnostic

SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0

One bounded remote NEW C3 BASELINE versus NEW H6 schedule comparison. The hypothesis follows the
flat original validation trajectory, not prior test errors. Both arms start at
seed 42 with the original rank 8, alpha 32, LR 0.0001 and dropout 0.1. The pinned
trainer's loss, accumulation and best-loss early stopping remain unchanged.

## Controller contract

`contract.json` freezes source hashes, original splits, settings and selection.
The historical replay in run 44441475865 failed. Its payload and trajectory
checks remain false in `historical-replay.json`, with original artifact hashes.
The old run omitted exact initial state, hash seed, order and hardware/thread
records. No concrete defect establishes how to recover that old numerical path.
This is a fresh within-job experiment, not a repaired historical replay.

Review additional campaign development-group exclusions before dispatch. The
workflow input accepts only a JSON array of normalized SHA-256 groups, at most
10000. An empty array means no additional groups beyond the original 1152.
Completeness of that list is a controller responsibility. This revision pins the
still-sealed audit and exclusions from the failed diagnostic. Changed exclusions
or audit hashes stop the run rather than replace the confirmation set.

`freeze_data.py` reproduces the original 768 train, 128 validation and 256 old-test
originals byte-for-byte. The old test contributes exclusions only. It selects a
new 256-original audit, 128 per label, from the complete pinned official test
file. NFKC/casefold/whitespace groups, contradictions and exclusions are resolved
before salted hash selection. Middle-range labels can invalidate a contradictory
group but are never selected. Insufficient source coverage fails without replacing
the audit. Source revisions, group IDs, exclusions and file hash are saved before
the first model comparison. Source text stays outside uploaded evidence.

Only train and validation paths reach `fit.py`. A process audit hook refuses old
test and fresh audit access. `stages.py select` sees validation evidence only.
The final process verifies the frozen candidate, config, transcript and audit
hashes. It reproduces validation margins before opening audit. An exclusive
audit-access marker prevents a second final measurement.

## Schedule and tracing

| Arm | Epoch cap | Steps per epoch | Maximum optimizer calls | Warmup calls |
|---|---:|---:|---:|---:|
| C3 | 3 | 48 | 144 | 8 |
| H6 | 6 | 48 | 288 | 17 |

Changing the epoch cap changes early warmup and decay rates. This is a schedule
comparison, not an identical-prefix continuation. The scheduler starts at LR zero.
The exact before-call LR list and actual optimizer telemetry are recorded. Full
16-pair accumulation windows are required. Ten-call gradient sampling records
finite post-clip L2 norms. Every window records actual label/style composition.

A hash-checked AST overlay compiles the original trainer function into a private
namespace. No shared file or original trainer globals are changed. The original
mixed validation evaluator runs normally once per epoch. Only added evaluations
and snapshot saves run in `torch.random.fork_rng(devices=[])`. Python and NumPy
states are restored too. Added full-format evaluations report both labels from
the same full-format batch pass, retaining n, correct, strict errors, ties, BT
loss and signed margin summaries. No additional loss or optimizer step is added.

Real remote Torch controls require identical fixed-step parameters with guarded
tracing. Deliberately unguarded DataLoader iterations must change both parameters
and RNG state. Local controls are fake/static and do not load Torch or models.
Saved snapshot reload must reproduce trainable tensors and validation margins.
Missing and unexpected keys are recorded and checked. Historical payload/loss
identity is no longer a gate for the new experiment and is never relabelled PASS.

`initial-state.json` captures exact named trainable bytes, Python/NumPy/Torch CPU
RNG, source file hashes, runtime versions and hardware/thread configuration.
One frozen untrained adapter initializes both arms. The build wrapper loads that
exact state and restores the captured post-build RNG. A before-training callback
requires identical boundaries and a fresh zero-step optimizer/scheduler.

Six explicit seed-42 training permutations are frozen before fitting. Both arms
use that same prefix with a separate DataLoader generator. Every epoch records
and verifies actual source-group order. This declared sampler change makes the
new baseline distinct from the old run. Added tracing remains RNG-neutral.
The selector requires both arm boundary matches, complete observed order and
restoration records before considering quality. No audit data reaches selection.

## Selection and quality

Every snapshot must pass all three validation formats: 80% overall, 70% for each
label, ten points over the length baseline and five points over the fixed
seed-matched untrained control. Upstream dataset and source shortcut gates stay
mandatory. Eligible snapshots rank by minimum of the six label/format accuracies,
worst-group BT loss, minimum format accuracy then earlier epoch and C3 before H6.
H6 improvement is reported only if its worst group gains five points without
reducing any group. This claim is separate from candidate eligibility.

No eligible candidate means diagnostic FAIL with audit unopened. Otherwise the
fixed candidate and fixed untrained adapter are measured once on all three audit
views with the same gates. These are 256 originals, not 768 independent samples.
Token IDs must equal the node tokenizer. Every complete rendered claim must fit
256 tokens. Missing conformance vectors fail. No quality bypass exists.

This unit emits an honest trained candidate. It does not export a node package or
claim parser acceptance, exported-runtime equivalence, repeated node score bits
or node qualification. Those remain release blockers for controller integration.
No absent live-class helper is treated as a passing preflight. A Torch quality
PASS is labelled `TRAINED_CANDIDATE` with `release_gate=INCOMPLETE`.

## Execution and failure evidence

The manual-only workflow requires the exact public repository and a standard
`ubuntu-24.04` GitHub-hosted CPU runner. Read-only permissions, no saved checkout
credentials, four threads and existing hash-locked dependencies are retained.
Dependency auditing must report zero vulnerabilities and zero skipped packages.
The workflow shares moderation-family concurrency with cancellation disabled.

The job ceiling is 330 minutes. The runner deadline is 290 minutes with a two-minute
preservation reserve. Each subprocess runs in its own bounded process group,
TERM followed by KILL after 20 seconds. Both fits share a 200-minute ceiling.
Setup, freeze, initial controls and audit have individual ceilings. No phase
reports PASS after a timeout. Partial traces and licensed snapshots are retained.

Epoch safetensors are stamped immediately with the full existing upstream grants.
The always-run preservation step also stamps the trainer's best disk checkpoint.
Tensor bytes are unchanged. Full checkpoint terms travel in the artifact itself.

## Source-only checks

```sh
bash verification/verify.sh
bash verification/moderation-diagnostic/verify.sh
actionlint -shellcheck= .github/workflows/evaluator-moderation-diagnostic.yml
```

Controller dispatches only after reviewing the patch. No model commands are
authorized locally. No dispatch, commit, push, credential access or service
change is part of this implementation unit.

## Limits

One seed and a narrow English toxicity proxy. No full policy taxonomy,
intermediate toxicity, identity subgroup, parent-context or multilingual claims.
No proof of freedom from foundation pretraining overlap. No established learning
root cause or six-epoch improvement. All neural, restoration and real RNG gates
remain unchecked until the remote workflow runs.
