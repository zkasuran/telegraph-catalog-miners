# Source-stage threat model

| Attack | Defence | Test or limit |
|---|---|---|
| Stock label or hypothesis shortcut | Balanced source hypotheses and frozen train-only prior | Prior failure retained; current real source gate passes at 50% |
| Cross-source leakage | Connected host/issuer/original groups and normalized evidence checks | Actual complete source manifests |
| Invented excerpt absence | No NotMentioned excerpt rows | Source constructor |
| Missing conditional evidence | Every annotated span retained; incomplete fragments excluded | Complete offset/index validation |
| Forged runtime identity | Literal bare hashes, not prefix-stripped comparisons | Three field mutation controls |
| Nonfinite score | uint32 and finite FP32 contract | NaN/Infinity and 500 random-bit controls |
| Local model operations | Remote-only guard and source-only gate | Denial tests |
| Fake quality pass | All format, label and improvement thresholds retained | Below-threshold failure controls |
| Failed validation accidentally exposes audit | Mandatory complete per-format/per-label validation gate | Runs 44612876956 and 44628637805 refused before audit; original reservation is unconsumed |
| Weak tag-only learning or loss-based selection | Direct source/complement claims, broader source groups and worst-format accuracy selection | Validation-only model selection; no audit gradient or threshold change |
| Slow or oversized HTTP input | Body/header byte ceilings, total input deadline and four connections | Actual local HTTP controls and remote identity/token controls |
| Wrong public model or report | Exact immutable URL/hash, full parser and reference vectors before serving | Serving preparation refuses mismatches |
| Unrestricted forwarding key | Exact existing common.lease_for with public proof hashes and pinned host | Only public restricted lease uploaded; ephemeral key stays on runner |
| Unchanged failed candidate presented as new | Deny all six observed failed tensor payloads | Exact actual failed package hashes retained; no restamp retry |
| Memorization or inadequate learned capacity | Unchanged gold training, retained rank growth and validation-only freezing | Actual failure traces retained; improvement remains remote-only and unverified |
| Auxiliary training changes protected source splits | Keep all 320 primary pairs and exact development/audit bytes; cap combined source groups | Actual reservation byte checks and authored auxiliary cap/overlap refusal |
| Rank expansion discards prior learned work | Preserve original factors, reward head bytes and alpha/rank scale; new output factors zero | Pure metadata/shape/refusal tests locally; actual tensor preservation remote only |
| Zero-padding is presented as learned capacity | Require actual nonzero new output factors after training | Zero/nonfinite count refusal locally; actual tensor counts remote only |
| Longer evidence contaminates development/audit | Wider citation policy is training-only; quote bounds and exact frozen rows remain unchanged | Explicit policy/hash/bound refusal tests and real split byte checks |
| A new training render hides a source shortcut | Gate the citation subgroup separately with the same thresholds | Actual complete and subgroup source shortcut reports |
| Development diagnostics consume audit rows or mismatched source answers | Explicit training/validation split checks, literal reconstruction and grouped overlap refusal | Authored audit-rename, contrast-mutation and overlap controls |
| Diagnostic scores are presented as a new model audit | Report supplied wire data as validation only, never emit registration-ready status | Literal prefix/count/nonfinite refusal and tie/format-disagreement controls |
| Validation capture reads protected audit rows or selects a model | Validation-only loader and frozen run/commit/source/package checks before reference execution | Authored audit-row and immutable-identity failure controls; actual scoring remains remote only |
| Reference capture hangs or silently accepts incomplete results | Capture deadline, per-request remaining-budget timeout, literal echoes and exact result count | Authored deadline/prefix/count/nonfinite controls and request-timeout refusal |
| Slow-drip reference response or connection failure defeats its budget/lock | Total request deadline includes lock wait, watchdog shuts down retained transport and finally releases the lock | Authored actual loopback slow-drip/healthy HTTP controls plus constructor-refusal and lock-budget controls |
| Partial engine/server startup leaks a process or log | Constructor readiness owns cleanup; serving creates server/thread inside its cleanup boundary | Authored launch/bind/thread-refusal controls and a healthy mocked startup control; no neural or SSH work locally |

Known limits: source-stage checks are not neural or reference-engine results. Document labels are not performance labels. The raw scorer uses the documented 512-token interface and complete source contexts up to 768 bytes; the distinct typed miner slot remains bounded to 256 bytes. No long on-chain contract transport or link dereference is claimed. Entity aliases and legal applicability require wider evidence. All 99 inspected audit groups are permanently retired. A link-only or bare verdict answer contains no clause evidence for this narrow task. Even a completed independent reference audit is not a canonical node win. No quality or identity gate may be bypassed to register.

## Opt-in grouped-format training

The paired-worst-format objective accepts training originals only. Each original's JSON/prose views must retain literal prompt, source hash, label and canonical answer reconstruction in a fixed adjacent order. Audit/validation rows, duplicate evidence, tampered answers, mixed batches and batches over eight originals are refused by paired_objective.py with source-only tests in test_paired_objective.py. Actual tensor operations in paired_objective_remote.py remain behind remote_only and require CPU, finite rewards and an even bounded shape. Actual gradient and refusal controls are remote-only and still unverified for this candidate. The default independent objective remains unchanged. Neither the scalar fixtures nor Torch gradient smoke prove reference-engine identity or model quality.

## Qualification-aware selection

quality_selection.py refuses missing labels/formats, nonfinite metrics and non-validation identities. It preserves actual trainer accuracy/loss fields and reports gate slack as a separate metric. The remote-only source adapter verifies the complete primary trainer hash before changing one metric expression and restores the original function on exit. Source-only tests cover aggregate-label masking and baseline constraints. The source gate compiles this exact adapter without importing neural packages. Actual model selection is unverified until remote execution. A bound validation trace is required before audit when this option is selected; no threshold or audit exclusion is relaxed.

## Synthetic source transfer

hans_curriculum.py accepts only the exact pinned original training file. The complete 30,000-row population is verified before balanced hypothesis selection; evaluation files cannot be renamed into it. The renderer distinguishes non-entailment from contradiction and excludes source template/heuristic metadata from model inputs. Authored tests exercise semantic, malformed-source, stock-label, quote, uniqueness and manifest-refusal controls. Frozen integer source probes choose the strongest trained direction; no floating-point refitting reproduces their bytes. Normalized overlap with legal source is refused and the curriculum remains auxiliary fitting, not audited capability. Full original MIT attribution/grants stay attached to every package and failed/successful checkpoint. Tensor training and quality measurement remain remote-only and unverified until actual execution.
