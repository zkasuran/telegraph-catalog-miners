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
