# Dated VAT observation consistency

This source collector retrieves a bounded spread of Norwegian AS company records
from Brønnøysundregistrene. It preserves organisation number, legal identity,
the declared VAT-register boolean and UTC retrieval time. Every minimal projection
is checked against its exact raw response hash and JSON pointer.

The target is conditional agreement with a trusted dated observation for
VENDOR_VERIFY. It does not establish universal vendor legitimacy, authenticity,
solvency or sanctions clearance. The experiment supplies observation context
explicitly. That context is not known to reach the deployed node.

## Actual result

The November 18 snapshot contains 512 distinct entities, 256 per observed VAT
state. Eight 64-row pages use offsets 448, 2752, 5824 and 8768 within each
state-filtered result set, sorted by organisation number. This is a dispersed
bounded slice of the accessible first 10000 results, not a population sample.
Fifteen same-registry exact-ID readbacks agreed. They are consistency checks,
not independent tax-authority corroboration.

Thirty-four entities declared a corporate group or lacked that field and were
excluded from diagnostic allocation. Shared normalized current/historical names
are excluded across distinct entities. Thirteen development-exposed IDs are
excluded, including the twelve-company sample with !FALSE. No development ID
occurred in the new slice. Undeclared ownership relations remain unknown.

The frozen snapshot diagnostic has 128 training, 32 validation and 64 closed audit
originals. Each split is balanced by VAT state. Variants count as the same original.
The final audit bytes are hash-checked only after freeze. No model was selected.
All observations are from one day. These splits are not a prospective time test.

The unchanged pinned kit data gate passed on 128 pairs: 100% unique rejected
answers, 50% length shortcut, zero negative leakage and 42.63% answer-only surface
probe. Exact source lookup and train-fitted TF-IDF overlap each ordered 100% of
training and validation pairs. **Training is blocked by the 65% overlap gate.**
The collector does not change facts or wording to hide that result.

A company-name token rule fitted on training scored 98.05% on its fitting set
and 62.5% on validation. The fitting-set number is not held-out performance.
Digits and alarm-keyword baselines scored 50% on both splits.

## Construction and controls

Each original has one source-supported assessment and one inverted assessment.
Only the assessment boolean changes. Organisation number, name and date remain
identical on both sides. Source facts remain unchanged and NLOD-attributed.
Both false-positive and false-negative assessment errors have equal counts.
These are constructed preferences, not official answers or human annotations.

The deterministic probe checks missing observations, formatted NO/MVA numbers,
misleading display names and complete real opposite-state evidence substitution.
A wrong source ID or date abstains. A misleading display name has no power to
change VAT membership for the exact ID. It does not certify that supplied name.
Deleted entities and missing VAT booleans are excluded, not assigned VAT false.
HTTP refusal or outage stops acquisition, not a negative source label.

All 160 training/validation originals passed each deterministic control. No
learned model was evaluated. JSON is the frozen preference format. Prose and bare
formats are available as repeated diagnostics, not extra source examples.
The largest inspected rendered request was 333 UTF-8 bytes. Exact token retention
with the pinned tokenizer remains unverified; bytes do not prove a token bound.

## Reproduce

Use Python 3.11 or newer. Collection and mechanics tests use the standard library.
The actual kit gate requires NumPy. No model dependency is imported or loaded.

```sh
bash verification/owners/vendor_verify/verify.sh
python3 verification/owners/vendor_verify/collect.py \
  --raw-root work/raw-cache \
  --out verification/owners/vendor_verify/new-snapshot \
  --exclude verification/owners/vendor_verify/example-exclusions.json
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  python3 -W error verification/owners/vendor_verify/probe.py \
  --data verification/owners/vendor_verify/new-snapshot \
  --raw-root work/raw-cache \
  --coefficients frozen-baselines.json --report new-source-result.json \
  --kit /path/to/telegraph-wasm-baseline
```

The probe returns 1 for blocked training and writes the requested new report,
never the retained original PROBE.json. First freeze coefficients once from
train.jsonl with baselines.py --train DATA/train.jsonl --out frozen-baselines.json.
All values are exact binary64 hex strings. Evaluation does not refit them.
Source acquisition
refuses existing output roots, unknown hosts, redirects, missing fields, checksum
errors, ignored filters, repeated IDs and raw-binding mismatches. Every request
has a 20-second timeout and 2 MiB response ceiling. Collection has a 600-second
elapsed ceiling. A refusal preserves its partial checkpoint. There is no retry
loop or paid fallback.

## Rights and limits

[B1] Brønnøysundregistrene primary documentation links NLOD 2.0 and defines
registrertIMvaregisteret as membership of Mva-registeret:
https://data.brreg.no/enhetsregisteret/api/dokumentasjon/no/index.html

[B2] NLOD 2.0 section 2 grants copying, changes and sharing for any purpose:
https://data.norge.no/nlod/no/2.0 . The full licence is attached to each acquisition.
Sections 3, 5 and 6 preserve exclusions, attribution and truthful representation.
Minimal company fields only reach the dataset. Transport responses are private
evidence including unneeded fields and are excluded from the patch.

Source grant SHA-256:
`c2c60474f811395722cedfe1580f5a5d0c4d2ccee1db4b2e2293490db74d02b5`.
Full source terms remain NLOD. SAND applies only to our additions. See NOTICE.

Remaining blockers: separate acquisition days, undeclared related-entity groups,
independent tax corroboration, actual token retention and trusted deployed
observation delivery. No useful learned advantage over perfect lookup is shown.
No training workflow or candidate checkpoint was added. No node qualification,
registry action, service change or remote dispatch was performed.


## Standalone source release

This directory is a source diagnostic, not a registration-ready neural adapter.
The exact source task is solved by lookup and the unchanged overlap gate blocks
training. No training workflow or model checkpoint is included. SOURCE-RESULT.json
records the actual source gate. The closed audit, raw responses and preference
rows are excluded from publication. Input/output paths are supplied explicitly.
Raw transport and projected output roots must be separate and new.

Frozen coefficients fit train.jsonl once and store exact binary64 hexadecimal
values. Reproduction loads those bytes rather than fitting on validation or audit.
See DATA-SOURCES.md, SECURITY.md, NOTICE and LICENSE for rights and boundaries.

## Current live-task boundary

The public prefix reports judgment while its sole proper descendant reports hybrid
and asks for EU VIES validity. Norwegian domestic VAT membership does not establish
intra-EU activation. TASK-SCOPE.json records only safe catalog facts. No live
question keys, raw taxpayer responses or exposure lists are published.

The source-check workflow is free public CPU with one ten-minute job. It runs the
same source gate and emits SOURCE_CHECKS_PASSED or SOURCE_CHECKS_FAILED with actual
run, commit and source hashes. Neither status means registration readiness.
No neural training, serving endpoint or registration is implemented by this source
contribution. Use your own exclusion list, not the empty example, for any new
acquisition. For a local primary-path smoke:

```sh
python3 -m pip install ruff==0.15.18
python3 verification/owners/vendor_verify/ci.py --out source-proof
```

## General preference development

The retained Norwegian observation diagnostic remains blocked for model training.
The added preference/ pipeline uses original licensed HelpSteer2 TRAIN comparisons
for general text-quality development. It passed the unchanged primary data gate
on 128 training originals with 64 validation originals. Another 64 originals are
sealed, not evaluated or claimed as an exact Vendor audit. The workflow train mode
runs only free public CPU for at most 350 minutes, selects using validation and
retains a licensed checkpoint. It cannot emit registration-ready evidence or
activate an endpoint. See preference/DATA-SOURCES.md and preference/SECURITY.md.

## Token-contract recovery

The first general development job passed its 128-pair dataset gate but failed
before training in its combined token-retention/conformance check. The repaired
pipeline measures all 384 train/validation rendered texts and emits aggregate
token loss and tokenizer disagreement counts before refusing a failed contract.
No audit is inspected, source split changed or token-retention gate lowered.
No model accuracy, package qualification or canonical registration is claimed.

## Training-capacity recovery

The completed diagnostic found one over-cap training text at 583 tokens.
All 128 validation texts fit within 455 tokens. Every capped training/node
vector agreed across all 384 texts. The new pipeline excludes only whole
over-cap training pairs before fitting, preserves validation and sealed audits
and reruns the unchanged dataset and token gates on retained training.
It freezes revised training-only baselines and requires improvement over both
original and revised controls. This is development, not a node qualification.
