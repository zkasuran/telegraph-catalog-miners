# LLM output evaluation owner pipeline

This trains a ModernBERT reward delta to score an evaluation of a supplied model output. It does not substitute assistant-answer preference training for the evaluator-of-output task.

## Source and task

HelpSteer2 by NVIDIA with Scale AI provides human helpfulness ratings over model-generated responses. All four public payloads are pinned, not just the first page of preferences. Original task/output strings must fit the published 256-byte input slots without cropping. Normalized prompt/output connected components stay together.

The fit set contains 160 balanced source originals. Development has 112 independent components. An unscored provider-validation audit is extended to 64 independent originals using unused source-train components. The allocation occurs before training. Neither audit model scores nor audit shortcut results select the candidate. The human helpfulness mapping to the five Telegraph rating names is a declared proxy, not universal factual ground truth.

Source text, source human annotations and our constructed evaluation preferences are distinct. Responses were model-generated. No human review of this new conversion or implementation is asserted. The JSON assessment is a written observation of the complete evaluated output, not a canonical processed field. The public scorer supports JSON explanations, prose explanations and bare rating strings. Every claimed format must pass independently.

## Literal qualification contract

The pinned primary trainer defines `render` and the current public evaluator guide specifies `### Task:\n` plus the preference prompt followed by `\n\n### Answer:\n` plus the answer. The public node supplies tokens for the finished scorer. The release gate executes that primary source expression against literal escaping and delimiter controls. The remote path confirms tokenizer conformance and complete visibility before training.

This is the public adapter qualification interface. The exact body selected from an arbitrary miner application is a different traffic-integration question and is not claimed here. No unpublished evaluator checkpoints or evaluation-pool rows are requested. Hidden node quality remains the node's decision.

## Verify and run

Run `bash verify.sh` from any working directory. It compiles source, validates JSON/YAML, runs adversarial and literal HTTP tests and checks a real training-source smoke. It imports no neural framework and needs no private evidence. HTTP unit tests use a labelled synthetic backend; the remote audit uses the real primary reference engine.

The owned workflow has evaluate and serve modes on bounded public ubuntu-24.04 CPU runners. Evaluate fits for at most three epochs, selects using development loss then runs a separate once-only audit. It preserves complete successful and failed checkpoint grants. The unchanged primary trainer data gate and primary package parser must pass.

The dependency preflight requires one exact release result for every locked package. Canonical names follow the Python Packaging name-normalization specification. Empty, partial, duplicate or substituted results cannot pass. Malformed or missing vulnerability data stays unknown, not zero. The declared torch CPU-wheel mapping remains limited to its pinned release-level advisory lookup and does not claim to vulnerability-scan custom wheel bytes. Ten source-only controls use synthetic auditor output; a real advisory scan still runs on the public CPU job before training.

The default json-original-v1 recipe retains three passes over the original JSON training pairs. New optional jobs use format-transfer-v2: JSON/prose views of all 160 original training groups, three bare-label views per rating and one fixed supplemental JSON serialization from an existing training original. Its 336 constructed views run for one epoch, below the original 480 example-presentation budget. These are not 336 independent groups or new human annotations. Validation/audit files and the original integer train-fitted baseline stay unchanged. The full primary repetition, leakage, length and fitted surface-probe gates still run remotely before training. No quality gain is claimed without measured validation and independent audit results.

The retained format-transfer-v1 constructor records the earlier 335-view experiment for provenance. The primary trainer updates only after complete groups of 16 micro-batches, so that earlier input left 15 terminal batch contributions unapplied. New jobs refuse an incomplete accumulation before reference/model preparation. Version 2 completes all 21 update windows without modifying the primary trainer, adding source truth or increasing the original budget. This is a source scheduling diagnosis, not an observed neural-quality gain.

Optional format-transfer-v3 retains the same 336-view budget but diversifies prose presentation. Two natural prose templates alternate by source-group identity, never by rating, length or model outcome. Both sides preserve the same rating/score, complete evaluated output and incorrect-rating choice. JSON and bare-label text are unchanged. This targets a measured development-source confound: the single prose template let a length-only preference score 70.98%. Legacy recipes/renderers and all original source files remain intact. No trained-model gain is claimed from removing a source shortcut.

Training saves the recipe and exact derived-view/source/baseline hashes. Exported metadata retains that provenance and the independent audit verifies those frozen identities before scoring. Workflow choices are closed, default to the retained recipe and are byte-identical to the source-gate copy. No local source test loads a model or runs the fitted surface probe.

The audit runs every claimed format through native Torch and the actual WASM reference engine. It checks literal bare-hash echoes, invalid identity refusal and repeated/fresh-process score bits. Registration-ready.json is written only after actual results reach 80% ordering, 70% for every rating slice, five points over untrained and ten points over the strongest shortcut. No shortcut above 65% is accepted. The report binds source, data, package, run, commit and runtime identities.

Audit rendering follows the frozen selected recipe and binds the exact rendered-view hash into the source-runtime manifest. Source text and annotations stay separate from these constructed presentation variants. No audit row or score selects the prose templates.

Training and audit compare complete native and pinned-node token arrays before scoring. No padding removal, input normalization or cropping can conceal a mismatch. The audit also keeps its run metadata callback separate from wire identity construction. Source-only controls execute the actual nested audit function with synthetic service doubles and reproduce the unfixed callback TypeError. These controls are not neural scores or reference-engine qualification.

Audit metrics refuse missing or unknown rating labels and nonfinite scores before computing any quality slices. Every required label must have observations. Ties count as losses. The source-only regression controls execute the actual pure metric function with synthetic label-only rows and numeric scores, never the frozen source audit or a local model.

Serving downloads only immutable verified candidate/report bytes, reproduces recorded reference vectors and uploads only a restricted public lease. The runner's ephemeral forwarding private key is not uploaded. The automatic action adapter handles proxy activation and registration. No paid runner or resource is used.

Serving validates the complete registration-ready report before fetching its model package or preparing a runtime/key. This includes pinned foundation/engine/tokenizer hashes, the independent source boundary, every supported format and rating label, unchanged quality margins and bounded finite reference vectors. Source-only controls execute the actual preflight prefix with synthetic path/fetch/env doubles. They do not create a model, key or lease and do not replace the activation adapter's independent checks.

Input-root regression vectors were generated by compiling the pinned primary Go input serializer and batch-root functions, not by duplicating the Python formula. Twelve batches include empty, repeated, reordered, 512-token and 100-item boundaries. Tests compare the exact canonical JSON bytes and hashes without sorting object keys or normalizing the wire. The actual primary Go client and HTTP handler also passed source-only interoperability controls through this frontend with a synthetic scorer. No foundation, adapter or neural engine ran in that harness. These input-serialization vectors are not the registration report's actual reference-engine score vectors.

## Limits and grants

Local source checks are not model quality. A source-held-out result is not a canonical win. The ordinal helpfulness proxy is not calibrated Telegraph factual truth and cannot verify information absent from the input. Source grouping does not prove absence from ModernBERT pretraining. The existing incumbent is compared only by the actual node, not by privileged access to its unpublished weights.

Source text and source annotations retain CC BY 4.0. ModernBERT retains Apache-2.0 and the primary kit retains MIT. New competitive additions use the reviewed SAND terms subject to those retained grants. Full terms are attached to successful exports and failed checkpoints. Read NOTICE, DATA-SOURCES.md and SECURITY.md. AI-assisted implementation used agentrouter/gpt-6.1-sol via OpenCode. No endorsement is implied.
