# Broader source-grounded product report scoring

This development mode tests whether a reward model can follow source status,
fact attribution and specimen scope across held-out response formulations.
It uses original licensed source facts, not model audit captures or invented
physical authenticity. JSON and prose both cover direct and typed request views.

The product field remains at most 256 UTF-8 bytes. A typed request can produce
a longer question. The pinned primary library accepted a 256-byte product and
rendered a 340-byte question. Its raw-answer fallback also preserved exact bytes.
This verifies the public library, not the deployed judgment caller's full path.

There are 1184 newly constructed training comparisons from 74 source records
and 1216 validation comparisons from 76 other source records. Sources, entities,
acquisition groups and formulation templates remain disjoint across the splits.
Source annotations and AI-assisted preferences remain separate. No human review
of constructed preferences is claimed.

Run the parent source-only `verify.sh` for syntax, static lint and all data tests.
Only the `develop_broad` mode on the standard public CPU workflow may load/train
the neural model or execute its reference engine. It preserves the primary
dataset gate, fits baseline coefficients on training only and selects by
validation loss. Actual tokenizer conformance and length checks happen remotely.

The earlier model's templates were solved perfectly by a snapshot grammar rule.
This development uses different formulations with held-out wording rather than
claiming that 62.5% was the strongest earlier shortcut. The new controls include
source-status copying, lexical overlap, length and immutable train-fitted
answer-only coefficients. Passing development is not an independent audit.

The independent 32-group audit remains unavailable. No registration-ready report,
active endpoint or canonical win is claimed. Original source facts retain their
grants. Our code/preferences use SAND and the pretrained model retains Apache-2.0.
