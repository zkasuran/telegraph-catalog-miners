# Score titles against the study they describe

This pipeline tests whether a reward adapter can distinguish a published title
for Study A from a title for Study B when both complete abstracts are present.
It preserves **224 independent source components** with author/ORCID/DOI and time
boundaries. It is useful narrow title-relevance coverage, not full factual
summarization proof.

## What is verified

| Item | State |
| --- | --- |
| Original grants and complete selected source | 448 primary articles reproduced without cropping |
| Source split | 128 train, 32 validation and 64 audit components |
| Rendering | 2,688 complete raw/JSON/prose inputs fit 256 tokens |
| Tokenizer | 88 primary conformance vectors passed |
| Frozen development controls | 50% for length, token length, TF-IDF overlap and answer-only counts in every format |
| Full upstream source gate | Passed on 256 training pairs |
| First model quality | Failed near chance, not registration-ready |
| First model integration | Full parser and actual engine agreed, repeat/identity controls passed |
| Revision 2 quality | Failed development at 50% in every format, clean audit unscored |
| Training-only diagnostic | Single requested-source control fitted eight train pairs; original two-reference control stayed at chance |
| Revision 3 quality | Failed development at 33/64 in every format; clean audit unscored |
| Matched batching diagnostic | Prepared on four train groups; no measured fit yet |
| Canonical node registration | Not attempted |

Titles are unchanged human-authored source associations. Donor preferences were
constructed by AI-assisted code. They are not human preference reviews. The
source is eLife at the revision and per-article URLs in `data/manifest.json`.

## Source-only local verification

```sh
bash verification/owners/text_summarization/verify.sh
```

The gate parses source and configuration, replays the fixed corpus contract,
checks negative controls and refuses local neural entry points. It does not
load weights. Optional full source/tokenizer replay uses the pinned kit and
source-only tokenizer dependency. The same gate runs in the public CPU workflow.

## Remote experiment

The stronger revision 2 recipe also failed development at 50% in every format
after 192 updates. Its development gate kept the fresh audit unscored.
The bounded diagnostic ran on four existing training groups. Both controls had
nonzero gradients and changed parameters. The original A/B control stayed at 50%
from step 16 through 128. The complete requested abstract control reached 100%
at step 16 and retained it. These are tiny-train fit scores, not independent
capability evidence. See [the actual diagnostic run](https://github.com/zkasuran/telegraph-catalog-miners/actions/runs/44713196605).

The first bounded run completed all three epochs and the actual reference engine.
Raw/JSON/prose accuracy was 48.44%, 49.22% and 50%. It did not improve the
seed-matched untrained control. The failed model is not registered. Its scored
audit is quarantined, not reused for later fitting or evaluation.

Revision 2 keeps the training and validation rows unchanged. It uses a stronger
fixed optimizer recipe and 64 fresh source/entity-independent audit groups from
the same licensed inventory. All development formats must pass quality and
baseline gains before audit scoring. This avoids consuming an audit on a model
that has not learned the development task. Full original grants are retained.

Revision 3 adds one complete requested-source warm-up epoch over the same 256
training pairs. Titles and abstracts are unchanged. Warm-up has 64 updates at
learning rate 0.0005, then a fresh optimizer runs the original three two-reference
epochs at 0.0001 with 192 updates. Rank 8, alpha 32, accumulation 4 and zero dropout
remain fixed. The warm-up checkpoint and its actual metrics are retained.
The actual run completed all 256 updates but selected development accuracy was
33/64 in every format. Warm-up mean loss was 0.125559; main mean loss was 0.724772.
The clean audit was not scored. See [the actual failed run](https://github.com/zkasuran/telegraph-catalog-miners/actions/runs/44778080472).

Before another candidate, `diagnostic_reciprocal` compares stochastic batches
with matched reciprocal A/B batches on four unchanged training groups. Both use
batch size 2, rank 8, learning rate 0.0001 and 512 updates from identical initial
parameters. Each control sees every training pair equally often. All prompts and
titles are the original complete two-reference task. The question is whether
balancing both targets within each update helps learn source selection. This is
training-only diagnosis, not independent quality or registration evidence.

The owned workflow uses Ubuntu 24.04, hash-locked CPU dependencies and explicit
timeouts. Fitting only receives the 256 training pairs. All three main epochs
complete. Validation loss selects a frozen checkpoint without advancing the
training RNG. Only then are the 64 independent audit groups scored in every
claimed format. A seed-matched untrained initialization and immutable train-fitted
shortcuts are measured separately.

The remote pipeline then runs the full current package parser against a fresh
live judgment-class snapshot and executes the exact exported package with the
pinned reference engine. It checks literal bare-hash echoes, finite score bits,
invalid-token refusals and repeated fixed vectors. `registration-ready.json`
is emitted only when the measured quality and runtime gates pass. A passing
source-held-out audit is not a canonical node win.

## Files and terms

`data.py` replays source boundaries. `primary.py` reproduces primary XML fields.
`baseline.py` replays immutable coefficients. `gates.py` runs source/tokenizer
checks. `curriculum.py` enforces the exact-train warm-up. `remote.py` owns training
and export. `runtime_check.py` owns reference
execution. `report.py` validates registration readiness. `SECURITY.md` records
failure controls and limits.

Source text and original title associations retain CC BY 4.0. ModernBERT retains
Apache-2.0. The Telegraph kit retains MIT. Our own competitive additions use
`LicenseRef-zkasuran-SAND-1.0`. Full grants and attribution remain attached to
unsuccessful checkpoints as well as any successful package. See `NOTICE` and
`DATA-SOURCES.md`.

`SAND-1.0.txt` retains the original reviewed own grant byte for byte. The LICENSE
scope appendix clarifies which source and failed-checkpoint portions retain their
own upstream grants. It does not change the SAND grant or restrict source rights.
