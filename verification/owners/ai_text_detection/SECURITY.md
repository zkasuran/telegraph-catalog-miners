# Threat model

| Attack | Defence | Test |
|---|---|---|
| Wrong source bytes or missing grant | Git blob plus SHA-256 and per-article CC BY clause | Manifest and tampered-XML controls |
| Sparse checkout omits gate inputs | Exact owned source and workflow paths in non-cone checkout | Incomplete-checkout refusal control |
| Further source growth replaces old provenance or exposes audit | Immutable parent bundle plus separate64-source manifest and fixed training dates | Source/entity/retired-author exclusions and secondary byte/role/overlap refusals |
| Additional data hides a failing old corpus | Separate unchanged752,880 and1008 primary data gates | Original data controls remain; fixture-only additive population tests |
| Source attribution exhausts checkpoint metadata | Compact complete grant/citation fields, full source sidecars | Attribution retention test; existing1MiB header ceiling remains |
| Secondary generator silently rerolls | Existing-output refusal and fixed source seeds/count/time caps | Local invocation refusal; actual generation/control results are remote-only |
| More bare training defeats repeated-negative controls | Two fixed whole secondary groups and unchanged actual/worst-case gates | Actual1008-row repeated-negative and worst-case<=0.10 controls |
| Format specialization biases an origin or rescales outputs | Weight16 symmetric on paired origins, actual mean normalization, runtime unchanged | Secondary weight balance test and all prior literal wire/quality controls |
| Training noise is presented as a proved quality cause | Fixed dropout-only hypothesis, unchanged data/loss/gate/inference contract | Policy and actual-hyperparameter source tests; actual repeat/quality controls remote-only |
| Dropout remains active or training differs from eval | Inspect actual modules and repeat one actual training vector before updates | Prepared native train/train/eval exact-score control, not node evidence |
| A fitted teacher reverses source labels or changes runtime scores | Positive bounded detached training target, original labels/losses retained | Scalar wrong-teacher/origin-swap/policy refusals; literal runtime unchanged |
| Teacher receives hidden gradient or nonfinite target | CPU FP32 batch-one shape/finite checks, detach,0.25-6 bound | Prepared remote value/gradient/shape/dtype/nonfinite/origin controls |
| Completed generation is rerolled for a better result | Immutable full data/producer/journal lineage restored before neural imports | Actual completed-generation SHA/prefix/source checks |
| Generator loses successful rows during recovery | Immutable39-group byte prefix with exact producer/source binding | Real retained-byte/grant/excerpt tests and producer corruption refusals |
| More tokens are presented as guaranteed word coverage | Word gate stays24; remaining budget fixed128 with actual64-prefix controls | Short-word refusal remains, token prefix has positive/negative controls |
| Failed generated candidates are represented as accepted preferences | Raw source/grant/token journal before cropping, separate accepted stream | Separate artifact names and acceptance gate; actual journal is remote-only |
| Capacity silently rescales learned deltas | Rank8/alpha32 to rank16/alpha64 preserves alpha/rank4 and all old slices | Strict policy and scalar controls; actual matrix delta checks run remotely |
| Resume discards learned rank16 capacity or mixes saved heads | Exact producer/selected-epoch/model/readout hashes and direct load | Actual opaque checkpoint/grant checks and new resume-policy corruption tests |
| Smaller step size silently weakens acceptance | Fixed lower-rate recipe, identical data/losses/selection/audit thresholds | Same all-format/label/shortcut refusal tests and literal wire controls |
| New columns overwrite prior learned state | Original slices/head retained, seeded extra A and zero extra B | Opaque descriptor/source controls; prepared remote key/shape/dtype/nonfinite/zero refusals |
| More parameters are claimed to work without learning | Actual extra B columns must receive finite encoder gradients before updates | Prepared remote primary-path capacity gradient control, not locally verified |
| Weak validation consumes an audit | All claimed validation formats must pass first | Validation refusal keeps audit sealed |
| Old audit tunes a later model | Retired articles and named authors quarantined | Fresh-audit source/entity exclusion test |
| Checkpoint state is swapped | Opaque SHA/grant checks locally, exact learned-state checks remotely | Wrong byte/source identity refusals |
| Repeated bare negatives leak | Fixed bare exposure stays within unchanged share/leak ceilings | Actual four-of-38 defect reproduction, unique-negative repair and larger-corpus bound |
| Restored validation data becomes training | Exact byte, producer and original source-role bindings | Locked-data role-swap refusal |
| Good average hides one failed format | Epoch selection prioritizes every actual gate and weakest format/label margin | Failed-bare/low-mean-loss selection refusal |
| Augmentation leaks held-out sources | New windows bind only original training IDs, spans and roles | Validation-role and span-mismatch refusal controls |
| More bare labels game the data gate | Larger actual corpus must pass primary gate and worst-case reuse bound | Expanded structural fixture under unchanged ceilings |
| Format loss weights bias origins or hide metric failure | Fixed bounded symmetric train-only weights; unchanged unweighted gates | Origin-asymmetry, nontraining-role and weight-ceiling refusals |
| Auxiliary objective changes labels, rows or runtime scores | Fixed symmetric correctness targets on existing train sides only | Wrong target, role and coefficient refusals; unchanged literal renderer |
| Invalid tensor shape or arithmetic silently poisons training | Batch-one CPU FP32 finite loss and gradients | Actual remote shape, margin, dtype and NaN refusal controls |
| Reused third-context outputs conceal a new generator operation | Exact original bytes and separate producer/source/plan hashes | Both-augmentation byte, plan and producer corruption refusals |
| Source byte length substitutes for origin discrimination | Mandatory train-fitted source-origin length baseline in every quality gate | High or omitted source-length baseline refusal |
| Auxiliary learns length or leaks audit rows | Paired equal-byte UTF-8 prefixes, source/entity/time splits and no audit fitting | Crop, normalized identity, role and immutable baseline controls |
| Training-only head enters the served package | Head is outside production model; expected 178 adapter tensor inventory | Actual remote parameter-exclusion and encoder-gradient checks |
| Harder-origin loss becomes an origin prior | Paired mean plus absolute counterpart gap is role-symmetric | Origin swap and bounded-loss controls |
| Group batching drops rows or changes accumulation silently | Exact view partition and eight literal rows per optimizer update | Actual 752-row/376-view population smoke |
| Readout resume changes or loses learned state | Exact source-bound checkpoint bytes and remote tensor equality | Model/readout byte and shape refusals |
| New training articles leak held-out entities | Fixed pre-2015 dates and existing/retired author/source exclusion | Supplement source/entity/date/grant refusals |
| Extra data hides a changed preference population | Explicit original752 plus supplementary128 counts and unchanged combined data gates | Original preservation, source-role, normalized overlap and repetition refusals |
| Supplement changes a weaker shortcut into the only comparator | Retain original train-fitted source-length rule beside expanded rule | Original78.125% validation shortcut still refuses85% overall |
| Supplement supervision fails to train the actual task head | Actual first supplementary batch must have finite nonzero reward-head and encoder gradients | Prepared remote primary-path gradient control, unverified locally |
| Composite source identity omits new grants | Bundle hashes both manifests; source attribution in package/checkpoints | Manifest and generated-byte bindings |
| Confidence penalty changes served scores or favors a label | Fixed symmetric training-only paired margin loss | Scalar symmetry and remote gradient controls |
| Old supplement is silently rerolled | Retained actual data, source bundle and original producer hashes | Locked-data byte/source checks and refusal to replace output |
| Compact cues fabricate facts or pad unique negatives | Exact first/last four source words on both sides, unchanged bare rows | Source-quote/prompt/label and no-padding tests |
| Compact layout weakens data checks or changes held-out wires | Train-only layout and unchanged repetition/quality gates | Real 752-row repetition smoke and validation-role refusal |
| First-page population claim | Complete pinned article tree, truncated=false | Inventory assertions |
| Article, named-author or text leakage | Whole-article groups, time blocks and normalized overlap removal | Cross-split corruption controls |
| Easy answer surface | Frozen train-only integer counts plus length and overlap probes | Symmetry and freeze tests |
| Invented human annotations | Separate journal-control assumption, synthetic origin and constructed preferences | Provenance assertions |
| Local neural execution | Remote guard before any model dependency or model download | Local invocation refusal |
| Wrong runtime package or hash prefix | Exact literal wire identity and fixed real engine vectors | Three identity negatives plus prefix control |
| Invalid tokens or nonfinite scores | Token-only bounded proxy and FP32 checks | Malformed-wire and token controls |
| Audit reroll or tune-on-test | Byte-frozen candidate before separate audit generation | Freeze and audit reuse refusal |
| Leaked serving key | Runner-only ephemeral key, restricted public lease artifact | Public manifest and upload allowlist checks |

Downloads have byte ceilings, HTTPS-only redirects and timeouts. XML entities are
rejected. Generator operations use pinned safetensors, no remote code and explicit
CPU limits. Saved pickle reward heads are generated by the same bounded job and
loaded with weights_only=True by the pinned exporter. They are never published.

Rank expansion is a disclosed architectural continuation, not a whole-checkpoint
byte equality claim. It retains the actual old rank slices and head. Actual
initial FP32 deltas are checked within absolute1e-6 before fitting; independent
runtime identity, repeat score bits and quality thresholds are not relaxed.
No rank16 model is loaded, constructed or executed locally.

Direct rank16 continuation retains every actually selected tensor, not just an
old rank8 prefix. The previous expansion proof remains historical evidence. The
new runner must prove exact full learned-state equality before fitting. The
selected-state artifacts keep their original grants; a failed quality gate cannot
be turned into an eligible registration by resuming or changing the learning rate.

Further training articles remain distinct from validation/audit sources and both
older training blocks. Generated rows are unbuilt until the real public CPU step.
Source-only generation fixtures are structural tests, never training material.
The original baseline and every quality/runtime gate remain. No additive source
publication is a model-quality or canonical-registration claim.

Source-margin consistency uses an unexported fitted readout only during legitimate
training. Its scores are not source truth, human review, calibrated probabilities
or an independent audit. This term never opens held-out audit rows or modifies a
served score. The old losses and all source/data/quality/runtime gates remain.

Zero training dropout does not waive any audit/runtime requirement or change a
registered package. Its effect is unverified until bounded public CPU measurement.
Base pooling and frozen prediction-head transforms match the pinned foundation
exporter and stay unchanged. Models are never loaded or evaluated locally.

Known limits: pretraining overlap is unknown. Journal controls do not certify an
individual's production history. DistilGPT2 is related to GPT-2. A source-held-out
pass cannot guarantee hidden node quality. Mixed and likely verdicts, calibrated
probabilities, non-English text and modern generators remain unvalidated. A five-
hour CPU serving lease is temporary and requires renewal without changing package
bytes or unresolved registration identity. Real engine and serving tests run only
remotely; source-only local tests do not pretend to prove them.

Continuation uses the exact saved adapter and a fresh bounded optimizer because
the original optimizer state was not exported. This is disclosed, not presented
as an exact training-process resume. Generator byte-fit diagnostics preserve
failures without retrying seeds for favourable outputs. Local checkpoint checks
interpret metadata and hash opaque bytes only, never load a neural model.

Layout augmentation is symmetric within each source pair and train-only. Literal
validation and audit strings stay unchanged. The training helper guards before
neural imports and emits actual epoch metrics without overwriting loss fields.

Additional source windows are generated only in a bounded public CPU step. All
source articles are reverified before model import. The job records actual
producer/run/commit identity, structural refusals and used view IDs. There is no
model-score selection of new windows. Starting-checkpoint validation is actually
measured and kept if later fitting worsens it. Source held-out audit data remains
sealed and original quality thresholds cannot be reduced by this pipeline.

Prior actual generated training data can be reused only with exact bytes,
producer/run/commit, source and plan bindings. This is not new data from an exposed
audit. Training loss weights never alter validation/audit scores or a pass gate.

The weight-32 follow-up keeps the existing weight ceiling and validates the exact
improved predecessor's licensed bytes. Historical failed checkpoints remain
available. A validation improvement still cannot authorize registration before
the fresh independent audit and actual reference-engine controls pass.

Further source contexts are checked against both original and earlier training
spans. Generation binds its separate source plan, run, commit and output hash.
Article/entity/time splits stay fixed. More bare contexts cannot relax the
primary data gate or the existing worst-case reuse ceiling; insufficient valid
data fails before fitting. Old trained input and every failed checkpoint remain
retained, not silently replaced by a source-only plan.

The correctness objective is a training-only hypothesis. Local tests exercise
scalar mathematics and input-policy refusals without loading a model. Actual
Torch value/gradient controls run in the public CPU job before model fitting.
Validation and audit still measure raw reward ordering in every claimed format.
No target smoothing or auxiliary loss makes a calibrated probability claim.

Native origin readout diagnostics are not node scores. Its source labels retain
the journal-control assumption and recorded generator provenance. Byte cropping
creates attributed source annotations, not new independent articles or certified
human review. Auxiliary quality is unverified until the remote job runs. The
actual runtime remains the unchanged production head over literal task/answer
tokens, checked by the full parser and reference engine before registration.

Coupled training uses one two-row source view per loader batch. Its source roles
are not separate contributor identities. The objective is symmetric under role
swaps, not a fixed class-weight correction. Both selected learned checkpoints
remain licensed and the auxiliary head remains outside the production adapter.
Source-coupled tensor gradients are remote controls, not independent quality or
node evidence. A native 81.25% aggregate can still fail per-label and shortcut
margins; those stricter checks are never replaced by the aggregate verdict.

Independent source growth does not authorize audit reuse or weaker thresholds.
The original preference rows and held-out source blocks are immutable. Additional
articles train source-origin features and explicit verdict preferences only after original-XML/grant verification
and actual fixed-seed generation. Their source-only optimizer updates are reported
separately, with the additional128 verdict pairs disclosed. Source bundle hashes, generator output hashes and original producer
identities remain distinct. A source-only fixture never becomes generated data.

Margin alignment cannot override any validation/audit failure. It leaves original
input strings and the public reference executor unchanged. Restored supplementary
source data retains its original producer and licence. New tensor controls and
model performance are remote-only claims, never inferred from local scalar tests.

Compact source quotation changes only fitted training responses. It does not
select source passages, create new contributors or alter evaluation answers.
Full upstream surface/quality checks must still pass. Original source and data
artifacts are kept so the layout can be distinguished from the unmodified corpus.
