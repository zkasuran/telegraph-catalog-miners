# Score archived station temperatures without guessing

This build checks a recorded daily high against an attributed NWS CF6 archive observation. A supported station and day get an exact Celsius error. Unknown locations, missing records and unsupported answer formats stay unknown.

It also prepares a bounded ModernBERT judgment experiment on the live WEATHER_FORECAST_VERIFY_GENERAL task. The neural experiment is not run or qualified. The archive scorer is a source-backed diagnostic, not a registered adapter or universal factual proof.

## What is built

- Strict GHCN parser with per-value source and QC admission.
- Immutable projection of 7,289 NWS source observations across 2,432 station-days.
- 128 training, 32 validation and 64 reserved audit originals with disjoint stations and calendar-month windows.
- Integer train-fitted length, overlap, numeric and answer-only controls.
- Literal task and response rendering tests plus adversarial input controls.
- A controller-only CPU workflow that freezes a package before a separate audit job.

Repeated response views are not extra independent examples. The two-degree wrong answers are constructed by AI-assisted code. They are not human preference annotations or natural miner responses.

## Try the useful diagnostic

Install the hash-locked non-neural lint tool then run verify.sh. The gate invokes the scorer against a real Miami September 1 observation, rejects a wrong number and rejects an ambiguous city alias. It also proves that the model runner refuses outside GitHub. No model is downloaded or loaded locally.

The score.py CLI accepts --location, --date and --answer. Locations are exact GHCN IDs or official station names from station-names.json. For example, MIAMI INTL AP is supported while the bare city Miami is not.

## Exact live task

The public schema asks for location and a past date. temp_max_c is marked scored, in Celsius with six decimal places. Low temperature, rain and condition are not extra factual evidence for the scored high. The scorer receives no trusted observation and no forecast issue-time.

The neural experiment uses that same request text without injecting a source passage. It may fail because date and place alone cannot establish arbitrary observed weather. A promising offline result would still require a fresh same-draw comparison against the incumbent and actual node qualification.

## Remote handoff

The installed action adapter publishes this pipeline under verification/owners/weather_forecast_verify and the exact evaluator-owner-weather_forecast_verify workflow. Jobs run only on standard public ubuntu-24.04 runners with read-only repository permissions. No wallet or provider credential enters the workflow.

Training is capped at 120 minutes. The training job is capped at 160 minutes and independent audit at 110 minutes. The existing hash-locked CPU model dependencies and reference kit are reused. Both trained and seed-matched untrained packages are audited through the actual reference engine, not native Torch scores. Failed checkpoints retain MIT, Apache-2.0 and SAND terms with unchanged tensor payloads.

Only a real passing audit can emit registration-ready.json. It binds the full parser, all formats, finite score bits and three real negative identity controls. A separate serving mode verifies those same vectors before emitting a restricted public forwarding lease. The forwarding key is ephemeral on the public runner and never uploaded. No signer is implemented here. The action adapter owns activation and gated registration.

## What is real and what is not checked

Real: complete six-station source inspection, item-level source flags, strict parsing, source-table construction, primary-path smoke and failure controls. The reserved source groups are station-days, not a claim that consecutive weather observations are statistically independent.

Not checked: neural training, exported model accuracy, actual tokenizer equality, reference-runtime neural execution, deployed node rendering, publication and same-draw incumbent replacement. The recipe's remote paths have local syntax and guard checks only.

The station archive is not a whole-city estimate. Archived CF6-derived values can be revised. The source contains no prior forecast. Trace rain is distinct from measured zero. Eight audit temperatures are negative Celsius but training and validation temperatures are not, a distribution risk the audit must retain.

## Terms and attribution

NWS source facts are public-domain material, not our copyrighted work. There is no NOAA/NWS endorsement. Source flags other than NWS CF6 flag 1 remain excluded. The complete mixed-source station files stay private, outside the public patch.

New code and trained additions use LicenseRef-zkasuran-SAND-1.0 under the reviewed repository LICENSE. Telegraph primary source remains MIT. ModernBERT remains Apache-2.0. DATA-SOURCES.md records the exact grants and retained boundaries.
