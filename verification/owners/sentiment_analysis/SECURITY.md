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
