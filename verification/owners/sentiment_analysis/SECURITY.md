# Frozen sentiment pipeline threat model

| Actor or attack | Defense | Test or implementation |
|---|---|---|
| Source replacement or truncated corpus | Hash-pinned archive, full three-file inventory and exact split hashes | source_data.inventory/prepare |
| Cross-split duplicates or changed original | NFKC, case folding, whitespace grouping and original split hashes | source_data.check_splits |
| Wrong model or metadata mutation | Frozen original SHA, one ignored recovery key, exact descriptor/payload/model-field comparison | contracts.recover_package/verify_recovery tests |
| Package parser defect | Unmodified pinned CheckLoRA with fresh hash-bound live classes | stagecheck.go and its Go source tests |
| Missing format or favorable aggregate | All three formats, 80 percent overall and 70 percent per label | contracts.validate_ready tests |
| Answer-only, length or overlap shortcut | Immutable training-only coefficients plus measured audit shortcuts | source_data.shortcuts and frozen-baseline.json |
| Wrong foundation, engine or tokenizer | Actual download/file SHA checks plus runtime identity echo | remote.assets/executor |
| Hash-prefix mismatch | Literal bare public adapter_hash, not a normalized echo | proxy and primary HTTP test |
| NaN, infinity or malformed result | Finite unsigned FP32 bits and exact positional result counts | contracts/proxy failure controls |
| Invalid identity or token request | Three actual HTTP identity refusals and invalid-token controls before scoring | remote.identity_controls |
| HTTP smuggling or overload | 1 MiB bodies/responses, 16 KiB headers, 8 connections, one scoring slot, no queue | pinned proxy harness |
| Hanging input or scorer | Absolute read and score timers, outer workflow timeout, bounded lease | proxy and serving lifecycle |
| Leaked evaluation tokens | No input logging; upstream executor output is discarded | remote.executor and serve.run |
| Credential leakage | Source gate blocks local model work; no provider/signer credential in source, artifacts or CI | remote_only and workflow |
| Stolen runner transport key | Ephemeral key, restricted listener, forced command, no local/Unix forwarding server policy, expiry | common.lease_for and installed action adapter |
| Wrong or stale audit artifact | Real run/commit/artifact/report hashes and exact serving bytes | fetch_report |
| Old audit reused after a quality loss | New corpus, fixed time boundary, no UCI audit or exposed node pairs used as data | retrain_data.grouped_select and exclusion test |
| Author, thread or normalized entity leakage | Remove every cross-time entity, select distinct originals even within each split | test_retrain grouped split failure controls |
| Constructed labels presented as human review | Preserve emotion annotations separately from valence mapping and Mixed juxtaposition provenance | retrain-plan.json and source manifest |
| Missing Neutral/Mixed measurements | Explicit four-label claims require all four per-label scores at the unchanged floor | test_retrain expanded-report controls |
| Local model loading during a source gate | Remote guards precede warm start, training, checkpoint export and audit imports | test_retrain local-neural-refusal controls |
| Failed checkpoint exported without grants | Embed original upstream grants and full GoEmotions Apache text at every best epoch | retrain.save_checkpoint; real export proof is remote-only |

Only the audit artifact's source-only metadata, fixed vectors and licensed
resulting delta are uploaded. Serving uploads only lease.json. Never upload
private keys, logs, received node inputs or private pool diagnostics.
Provider and approved Telegraph signer credentials stay in the installed
automatic adapter, never with the owner or runner.

## Known limits

The public endpoint is a bounded integration lease, not paid ongoing hosting.
Untrusted clients can consume its bounded scoring slot. The node client has
positional results, not per-result ids; equal score bits can be legitimate.
Pinning the real executor and validating input roots protect the intended order.
The dedicated SSH account's remote-only policy is independently checked by the
installed adapter. A public key restriction is not a claim that this policy was
already installed. SHA binding is not a signature from the source author.

Source labels are binary annotations, not full sentiment truth. The model is
fitted public training material. The old exposed source audit is never used for
adaptive selection. A genuine quality loss requires a better development model
and a clean independent audit, not another random registration of the same bytes.

The new source is an English emotion-to-valence proxy. Mixed texts are constructed
juxtapositions, not original human Mixed annotations. All new neural quality is
unverified until the bounded remote jobs complete. Validation may select a frozen
epoch; its audit cannot tune that epoch. Raw Reddit authors are used only as hashed
grouping keys and are never included in the public manifest. Reason correctness
and numeric confidence remain outside the independently tested coverage.

## V3 continuation controls

| Risk | Defense | Test or implementation |
|---|---|---|
| Failed checkpoint presented as passing | Explicit failed filename, unchanged byte hash and original embedded grants | test_v3 frozen checkpoint control |
| Exposed audit recycled into later learning | Quarantine every old audit source, author, thread and normalized original | v3_data.select_groups/check_groups and refusal tests |
| Mixed solved by composition artifacts | Composition controls in all labels, varied connectors/order and train-fitted prompt shortcut histograms | v3_data.pairs/baselines and composition tests |
| Aggregate validation hides a failing format or label | Measure all three formats and all four labels; select weakest-label performance first | v3_remote.selection_key and missing/weak-label controls |
| Nonfinite or incomplete validation report | Exact label/format inventory and finite metric validation | test_v3 invalid metric controls |

The v2 independent audit failed and is not reused as a new audit. V3 source
labels still measure an emotion-valence proxy. Stronger rater agreement is not
universal factual proof. No new neural quality or canonical win is claimed before
the exact remote results and fresh node decision pass their unchanged gates.

## Passing artifact release gate

The immutable model, ready report and source manifest are byte-bound in the source
gate. test_v3_release.py exercises the actual report and rejects altered hashes,
missing formats/labels, below-floor accuracy, excessive shortcut baselines, false
reference-runtime claims and nonfinite score bits. The exact model keeps every
embedded upstream grant. No release test fits on audit data or loads a model.

The published source audit passed. Live endpoint, receipt and canonical node
qualification remain separate gates. The served executor source must match the
audited serving_commit, not a modified runtime with a familiar model filename.
