# Recover the trained sentiment evaluator

This workflow recovers the unchanged package from public run 44266324976.
It does not train a model. Original training identity remains separate from
the recovery run and commit.

## Frozen input

| Input | Binding |
|---|---|
| Repository | zkasuran/telegraph-catalog-miners |
| Training commit | a3a6f86c1bd831b41de8d99d12b0620c3a287f6a |
| Original artifact | 14413342566, training-SENTIMENT_ANALYSIS |
| Artifact ZIP SHA-256 | b647e50c448fe42431a84da0509a9f0ed119f618f0786435c9a93b14048ed32b |
| Package SHA-256 | f2eefcc7b7fd807bcbd260d97c8b3b0a6b50276199b475131df968a5b60d43d9 |
| Quality report SHA-256 | 4ffc26072a347b9af879afbd3c17fbecc1c1115bf6fe141943088577dff23422 |

The original artifact expires on November 25, 2026 at 09:09:09 UTC. Missing or
expired artifacts fail. There is no alternate artifact or retraining fallback.

## Remote job

Manually dispatch `.github/workflows/evaluator-sentiment-recover.yml` after the
controller integrates this patch. It runs on standard ubuntu-24.04 CPU with a
180-minute job ceiling. The runtime has a 7,200-second internal ceiling and an
outer process timeout. Artifact retrieval uses only the ephemeral workflow
token with actions:read. The token is scoped to that step and is not forwarded
to the signed archive download URL.

The helper verifies original run identity, commit, artifact id, archive digest
and required member hashes. It regenerates the exact source-only splits with
the original source generator and checks all split hashes, balanced counts and
normalized group exclusion. Pinned foundation, engine and tokenizer assets are
hash-checked before the executor starts. No base-model weights are fetched.

Immediately before Stage 1 parsing, fresh bounded HTTPS reads capture exactly:

- `https://devnode.telegraphprotocol.com/api/wasm?resolve=SENTIMENT_ANALYSIS`
- `https://devnode.telegraphprotocol.com/engine/v1/intents/SENTIMENT_ANALYSIS_GENERAL`

The snapshot contains the retrieval time and raw responses. `packagecheck.go`
requires its external SHA-256, exact URLs, identities, class agreement and all
three child schema family pointers. The parent's scorer fallback may be GENERIC.
It does not override the child's schema family. No class override or model-load
flag exists in the checker. The unmodified pinned `CheckLoRA` retains all Stage 1
checks with the original RequireTraining=false semantics plus an exact selected
child check. Static acceptance means loaded=false.

`runtime_all_formats.py` reconstructs JSON, prose and bare-label views from the
same 256 Yelp originals, 128 Positive and 128 Negative. It scores all 512 vectors
per format twice and requires identical repeated FP32 bits for every vector.
Scores must be finite and every response must match the package, foundation and
tokenizer identities. Each format must clear 80 percent overall, 70 percent per
label, ten points above its length baseline and five points above the original
seed-matched untrained accuracy. The latter is a preserved PyTorch baseline,
not a new untrained-runtime measurement. Ties are failures.

## Output contract

Successful output is the `sentiment-recovery-candidate` Actions artifact.
`candidate-manifest.json` has schema_version=1 and status
`CANDIDATE_PENDING_CONTROLLER_REGISTRATION_TEST`. It names the unchanged package
file, SHA-256, checked Keccak-256 and size. It includes separate training_identity
and recovery_identity objects, kit revision, pinned runtime inputs, all format
metrics and hashes of the evidence files.

Other required files are `result.json` (unchanged original quality report),
`artifact-binding.json`, `data-binding.json`, `class-snapshot.json`,
`foundation.json`, `package-manifest.json`, `runtime-result.json` and
`runtime-score-bits.json`. Every format in the score evidence has score_bits,
repeated_score_bits and token_vectors_sha256. The candidate emitter recomputes
all quality gates from the retained scores before emitting the manifest.

Validator stdout and stderr persist in `package-check.stdout.txt` and
`package-check.stderr.txt`. `package-check-status.json` records the exit code,
phase and diagnostic truncation separately from quality PASS. Runtime logs are
in `runtime.log`. Failure output includes `recovery-error.json` when a recovery
helper phase fails. Upload runs even after a job step fails.

The recovery artifact is a controller integration candidate. Registration,
endpoint leases, signing and node qualification are explicitly not attempted.
Neutral, Mixed, multilingual output, reason quality, backbone pretraining overlap
and population accuracy are not validated by these source-only tests.

## Local static checks

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error -m unittest discover -s verification/training -p 'test_*.py' -v
ruff check --no-cache verification/training/sentiment_recover.py verification/training/runtime_all_formats.py verification/training/static_sentiment.py verification/training/test_sentiment_recovery.py
/tmp/opencode/actionlint -shellcheck= .github/workflows/evaluator-sentiment-recover.yml
```

Build `packagecheck.go` from the pinned kit directory with Go 1.25.7 using
`go build -mod=readonly -trimpath`. The static CLI below takes controller-owned
scratch paths and never loads a foundation. It captures fresh public classes,
checks the real original package and runs parser mutations with updated outer
hashes so the parser itself must refuse invalid packages.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error verification/training/static_sentiment.py --checker CHECKER --package ORIGINAL_PACKAGE --out PRIVATE_SCRATCH
```

## Source contract

The primary implementation is pinned to kit revision
`1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd`:

- [CHECK] Telegraph `pkg/adapterbuild/lora.go`, the unchanged CheckLoRA interface and all Stage 1 checks.
- [CLASS] Telegraph `pkg/scoring/catalog/classes_db.go`, checked UseClasses installation.
- [FORMAT] Telegraph `float_runtime/export_adapter.py`, foundation target dimensions.
- [DATA] Original `verification/training/data.py` at the frozen training commit, preference templates and source selection.

The helper hash-pins [DATA], sources.json, requirements.lock and licences.json.
Terms remain embedded in the original package and checkpoint bytes. Helper
source uses LicenseRef-zkasuran-SAND-1.0 under the repository's existing terms.
