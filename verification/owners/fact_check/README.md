# Check capital claims against frozen source statements

This FACT_CHECK owner pipeline trains a narrowly scoped reward model on primary
CC0 Wikidata P36 assertions. It checks claim consistency with those statements,
not universal factual truth or a live lookup from an opaque identifier.

The preserved pilot has 128 training, 32 development and 64 independent audit
originals. Entity/relation groups and normalized text stay disjoint. All labels
and candidate errors are transparent source-bound constructions, not human
preference labels. Full source, licence, revision, reference, qualifier and
accepted-alternative bindings remain in the frozen artifacts.

## Verify without loading a model

```sh
bash verification/owners/fact_check/verify.sh
```

This command uses only source contracts and failure controls. It checks the
original grants, exact archived primary bytes, grouped splits, complete recorded
256-token input bindings and the fixed three-epoch recipe. The source gate does
not load a model or score the audit. Run the actual tokenizer/data gate in the
locked CPU environment; `gates.py` preserves its 88 conformance-vector checks
and immutable train-fitted diagnostic coefficients without refitting for byte
equality on another machine.

## Bounded public CPU workflow

The owned workflow is `evaluator-owner-fact_check.yml`. The preserved `train` mode
reproduces the original experiment. It uses the hash-locked dependencies, a fresh
seed-matched initialization and all three epochs. Development loss alone selects
the checkpoint; the audit is measured only after selection. The thresholds remain
80 percent overall, 70 percent per label, five points over untrained and ten
points over the strongest frozen shortcut. No shortcut may exceed 65 percent.

The workflow executes the full live-class package parser and pinned reference
engine. It records actual finite score bits, repeated/fresh-process controls and
three negative identities. `registration-ready.json` is emitted only from real
passing audit/runtime artifacts. Native Torch logits are not substituted for
reference-engine evidence.

The original run 44674965147 failed at 32 of 64 audit pairs. Its False-labelled
pairs scored zero correct and its validation showed the same True preference.
The package parser passed and reference execution agreed with native ordering.
That package is not registered. Its exposed audit cannot be reused.

The successor `develop` mode first checks the full primary discovery inventory.
It then fits only on the original training split for a predeclared 24 epochs,
with learning rate 0.001, accumulation four and no LoRA dropout. Only development
loss selects a checkpoint. The original frozen bytes and quality thresholds
are unchanged. Development outputs are explicitly not registration-ready.
Even a passing development result needs a clean independent audit, full parser
and reference-engine qualification before activation.

The `serve` mode refuses any package or report that differs from those audited
bytes. It reproduces the actual reference vectors before exposing a token-only
proxy and uploads only the restricted public forwarding lease. The installed
automatic action adapter controls source publication, endpoint activation and
the existing selected Telegraph signer. No provider or signer key reaches this
pipeline. No source-only pass or receipt is called a canonical win.

## Rights and limits

See `DATA-SOURCES.md`, `NOTICE`, `SOURCE-CC0.txt`, `KIT-LICENSE` and
`UPSTREAM-APACHE-2.0.txt`. Original grants remain attached to failed checkpoints
as well as successful exports. Our code and fitted additions use `LICENSE`.

Multiple applicable capitals, unsupported qualifiers, unknown values and missing
references are excluded, never relabelled False. Missing evidence belongs to
Unverified and the other graded verdicts require broader reviewed coverage.
The unqualified source statements are scoped to their pinned revision, not a
claim that the observed world cannot change. Actual trusted observation delivery
through the deployed judgment caller remains unverified.
