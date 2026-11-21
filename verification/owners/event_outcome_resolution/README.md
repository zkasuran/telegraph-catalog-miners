# Settle dated event questions from supplied award evidence

This pipeline trains a reward adapter that separates yes, no and unresolved
for historical award-recipient questions. It uses complete official recipient
lists with exact event editions and dated cutoff rules. An unknown result is not
a negative result. It cannot predict an opaque future event or authenticate an
arbitrary caller's evidence.

## Source and independence

All 688 API-reported official award records were acquired across seven pages.
Original JSON and source hashes are retained. The 2022 economics record says
`dateAwarded: 2011-10-10`; it is quarantined without rewriting the source.
Missing recipient lists supply no negative labels.

The full derived corpus keeps 203 training events through 1950, 94 validation
events from 1951-1970 and 331 audit events from 1971 onward. Repeated recipients
that cross those time boundaries are excluded in all affected editions. All
counterfactual questions for an event stay together. No normalized person, event
or prompt crosses a split.

The frozen CPU subset contains 128/32/64 events and 768/192/384 preference
pairs. The audit has **39 independent event/entity connected components**, not
64 independent groups. This distinction was checked before any model operation.
The earlier source-only split had only 16 independent audit components and was
preserved rather than treated as passing the new gate. No neural audit was run
on that earlier split.

Length, token overlap and an immutable train-fitted additive answer-text baseline
each measure 50% on every frozen split. The unmodified upstream surface gate
must also pass remotely. These probes do not rule out every shortcut. JSON and
prose are repeated response views of the same independent source groups.

## One local command

```sh
bash verification/owners/event_outcome_resolution/verify.sh
```

The gate uses only the standard Python library. It compiles source, validates the
owned workflow, rechecks source grants, rebuilds byte-identical data and tests the
primary scorer and literal batch wire. It rejects missing evidence, conflicting
booleans, Unicode name confusions, source corruption and insufficient independent
groups. It detects an actual unknown-to-no scorer mutation. No neural import,
training, model load or model inference occurs locally.

## Complete bounded remote path

The exact owned workflow runs only on standard public `ubuntu-24.04` CPU jobs.
It uses hash-locked dependencies, read-only repository permissions and at most
two concurrent engine-audit jobs. No paid or self-hosted runner is used.

Live class and routing are checked before frozen rows or neural assets are opened.
A mismatched child class remains a refusal, not an override. Failed preflights
retain bounded class identities and response hashes with the checkpoint terms.

1. Train on training rows and select the unchanged upstream best validation
   epoch. Export the frozen candidate and its seed-matched untrained package.
   Training never reads the audit rows. Failed checkpoints retain all terms.
2. Audit each claimed response format once with the actual pinned foundation
   WASM engine. Score both frozen packages, check the full current parser against
   a fresh live class snapshot and measure per-label accuracy and shortcuts.
   No native Torch audit substitutes for the node's engine.
3. Require 80% overall, 70% per label, five points over untrained and ten points
   over the strongest frozen shortcut. Record eight real fixed token/score-bit
   vectors, repeated bits and three actual negative identity controls.
4. Combine only passing measured reports into bounded `registration-ready.json`.
    Every report binds run, commit, source, package, engine and executor identities.
    Report containers and hashes are type-checked. Boolean values cannot stand in
    for numeric proof fields. The accuracy and baseline-improvement floors stay
    unchanged.
5. Serving mode rechecks the exact immutable package and report, runs the same
   engine and emits only an expiring restricted public forwarding lease. The
   automatic controller adapter owns activation and the Telegraph signer.

The public namespace is standalone. It does not import sibling owner code or
require private working files. No workflow dispatch is performed by the source
gate. Publication, registration and canonical node decisions remain separate
observations, never inferred from a prepared pipeline.

## Real and still unverified

### Explicit parent-only source plan

```sh
python3 verification/owners/event_outcome_resolution/parent_scope.py --out parent-only-plan.json
```

This source-only command checks the exact parent judgment class, parent-served v4
wire and parent material status. The plan claims only EVENT_OUTCOME_RESOLUTION,
not GENERAL. It binds the unchanged frozen split hashes and leaves validation
accuracy unset. A training record can be built only after a finite validation
measurement exists. Preparing the plan neither trains a model nor registers it.

Parent material must meet the stricter of the node's contributor minimum and
five contributors. Child eligibility never substitutes for the parent. The
parent-only path is selected by workflow mode `verify_parent` (or `serve_parent`
for the verified endpoint). This keeps GitHub's ten-input dispatch limit. Training
exports only the parent training-record intent. Candidate, audit, report and
serving all bind the same explicit scope. The actual static parser checks the
parent v4 wire without treating GENERAL as judgment. Existing general-mode class
checks are unchanged. Actual model and full production qualification results
remain unmeasured until the bounded remote workflow runs.

If quality/runtime proofs pass but parent material is not ready, the report step
preserves `material-condition.json` and refuses `registration-ready.json`.
Its failed result artifact retains the licence notices. Failed quality or runtime
proofs never become a material-only wait.

The report stage verifies the exact candidate package filename, count and digest
before it emits readiness. It copies the verified byte buffer before the report,
not a later file read. Changed or missing package bytes cannot leave a ready
report behind in a failed job.

| Item | State |
|---|---|
| Complete official source inventory and original grants | Captured and hash-bound |
| Evidence-scoped diagnostic and source-only wire controls | Locally tested |
| Frozen source groups and integer baseline coefficients | Reproducible |
| Neural reward quality | Requires actual remote job |
| Full parser and real reference-engine results | Required remote gates |
| Serving lease and public endpoint | Only after verified package |
| Fresh canonical registration | Not claimed by this source |

The declared child wire transports only `event`, at most 256 UTF-8 bytes. Winner,
sources and resolved time are off-chain. The supplied evidence fits that actual
string contract. `resolved_yes: false` is not interchangeable with outcome no.
Literal judgment rendering preserves whitespace. The runnable endpoint refuses
prefixed or altered hash echoes and accepts only bounded token arrays and finite
uint32 FP32 score bits.

## Terms

Official data is from Nobel Prize Outreach under CC0 with retained implementation
terms. No endorsement is implied. Source records remain separate from
AI-assisted constructed preference answers. No human preference review is claimed.
New additions use `LicenseRef-zkasuran-SAND-1.0`. The original MIT reference kit
and Apache-2.0 ModernBERT grants remain attached in source, packages and checkpoints.
See LICENSE, NOTICE, DATA-SOURCES.md and SECURITY.md.
