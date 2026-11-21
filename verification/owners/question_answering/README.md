# QUESTION_ANSWERING source preference evaluator

This pipeline ranks compact English single-turn answers for source-preference
relevance and usefulness. It does not certify factual truth. The first 128-pair
model failed quality at 24/64 despite correct reference-runtime plumbing.
That exposed audit is retained as failed diagnostic evidence and excluded here.

Generation two uses 768 training and 128 development source components. Every
one of the prior 224 exposed source components and matching normalized actual
inputs is excluded. The source files were completely inventoried. One original
human sibling preference survives per connected tree component. No negatives
are fabricated and no differing answer claim is cropped at the 256-token cap.

The development source gate passed before the design and train-fitted baseline
froze. Its strongest development shortcut is 60.9375%, below the unchanged 65%
ceiling. A separate 64-component audit was constructed only after that freeze.
Source-held-out grouping is not an independent node win.

Run the standalone source gate with `bash verify.sh` from this directory or pass
its path from a public checkout. It uses only the standard library and imports
no neural framework. The remote workflow owns all neural work on a standard
public Ubuntu CPU runner, four threads, rank 8, alpha 32, 256 tokens and three
fixed epochs. Development loss alone selects the checkpoint; its evaluation
RNG is isolated. If development ordering misses the quality floor, audit is not
scored. Data, model, parser and engine failures are retained with complete terms.

The remote package checker runs the unmodified full primary parser after an
exact live family/child class preflight. The reference engine scores the same
complete audit inputs. Literal bare hash echoes, finite score bits, repeated
bits, fresh-process controls and wrong-identity refusals are checked. A ready
report is emitted only from actual passing results. Native Torch output is not
assumed to be node output. All quality and baseline-improvement floors remain.

The workflow has qualification and exact restricted-serving modes. Serving
downloads only immutable proof-bound public bytes, verifies recorded engine
vectors and publishes only its restricted public lease. Signer and provider
credentials never enter this code. The automatic action adapter owns actual
publication and testnet registration. Matching fresh canonical owner, ID and
package hash is required for a win.

Our competitive code and trained additions use LicenseRef-zkasuran-SAND-1.0.
Original source text, annotations and ModernBERT/Telegraph portions keep their
unchanged Apache/MIT grants and notices. See DATA-SOURCES.md and SECURITY.md.
No human code review or licensor endorsement is claimed.

## Development-only repair after the expanded-data run

Run 44675306068 completed 144 optimizer steps but was rejected on development:
the selected epoch ordered 80/128 correctly, with 55/64 longer-preferred versus
25/64 shorter-preferred cases. The audit was not scored. All original data and
frozen baseline bytes remain unchanged; that audit remains unconsumed.

The previous fixed training objective was Bradley-Terry loss plus 0.25 times the
squared mean of reward margin multiplied by source length direction. Every
optimizer group contains eight source longer-preferred and eight shorter-preferred
pairs, without changing or repeating any source preference. A detached first
pass measures the moment; a second pass replays the same dropout RNG and supplies
the exact moment derivative with one-pair activation memory. Source-only finite
differences verify that gradient. This is a training-only nuisance regularizer,
not a serving-time score rescale. Real neural behavior and improvement remain
measured only by a bounded remote job. Development loss still selects the
checkpoint, with all audit, quality and baseline-improvement gates unchanged.

## Prompt-conditional candidate after the debias loss

Run 44855724690 completed all 144 steps but ordered only 81/128 development
pairs correctly. Its selected epoch had 49/64 longer-preferred and 32/64
shorter-preferred wins. The length repair gained one net win, not a quality pass.
The audit remained unused. The next mechanism keeps original Bradley-Terry
supervision and adds conditional ranking against a displaced training prompt.

All 768 auxiliary mappings use training groups only, retain unchanged source
texts and fit complete inputs within 256 tokens. Their displaced-margin neutrality
constraint is constructed, not human-reviewed or source truth. This trains prompt
dependence without an inference-time subtraction or score remap. Remote diagnostics
measure original, answer-only and displaced-prompt margins on development, with
real ties and token counts. They are not an independent audit and do not select
the checkpoint. Development loss still selects; the original audit and all gates
remain unchanged. Model improvement is unverified until the next remote result.
