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
| Neural quality and real engine | Must be measured remotely, no result claimed here |
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

The owned workflow uses Ubuntu 24.04, hash-locked CPU dependencies and explicit
timeouts. Train only receives the 256 training pairs. All three fixed epochs
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
checks. `remote.py` owns training and export. `runtime_check.py` owns reference
execution. `report.py` validates registration readiness. `SECURITY.md` records
failure controls and limits.

Source text and original title associations retain CC BY 4.0. ModernBERT retains
Apache-2.0. The Telegraph kit retains MIT. Our own competitive additions use
`LicenseRef-zkasuran-SAND-1.0`. Full grants and attribution remain attached to
unsuccessful checkpoints as well as any successful package. See `NOTICE` and
`DATA-SOURCES.md`.
