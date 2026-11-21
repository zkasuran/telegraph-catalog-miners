# Development input grants and lineage

## [W1] NWS facts

Primary grant: https://www.weather.gov/disclaimer.

> The information on National Weather Service (NWS) Web pages are in the public domain, unless specifically noted otherwise, and may be used without charge for any lawful purpose so long as you do not: 1) claim it is your own (e.g., by claiming copyright for NWS information -- see below), 2) use it in a manner that implies an endorsement or affiliation with NOAA/NWS, or 3) modify its content and then present it as official government material.

NWS facts are not subject to our copyright. No endorsement or official climate
certification is implied. Third-party origins do not inherit this grant.

## [W2] GHCN source semantics

Primary source: https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt,
Version 3.35. Source flag 1 means:

> CF6 (form F6) daily climate summaries from the U.S. National Weather Service

Only unchanged flag-1 TMAX cells with blank measurement and quality flags enter
this development projection. Each row retains the original monthly header,
eight-byte cell, station/date, line number, source URL and full capture hash.
The full mixed-source captures are not published or used as training rows.
The new source manifest records all seven relevant station captures, not just
the four supplemental stations. The facts are archived observation-day values,
not prior forecasts or UTC-instant estimates.

Citation: Menne et al. (2012), GHCN-Daily Version 3, NOAA National Climatic Data
Center, https://doi.org/10.7289/V5D21VHZ. Source capture dates are in the manifest.

## [K1] Primary trainer

Telegraph kit revision 1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd,
copyright 2026 telegraphprotocol, MIT under the family KIT-LICENSE.
Its grant permits use, copying and modification subject to retaining that notice.
The upstream trainer, tokenizer conformance and parser gates are not bypassed.

## [M1] Foundation model

Answer.AI and LightOn ModernBERT-base revision
8949b909ec900327062f0ebf497f51aef5e6f0c8. Its primary card states:

> We release the ModernBERT model architectures, model weights, training codebase under the Apache 2.0 license.

Full terms remain in the family UPSTREAM-APACHE-2.0.txt and all saved checkpoints.
No model weights are downloaded, loaded or executed by the local source gate.

## Construction and fitting

Preference answers are deterministic AI-assisted constructions, not human-reviewed
votes or natural miner responses. Rotating answer views do not create extra source
groups. New additions use the family SAND terms without restricting upstream facts.
Fitting and model work run only in bounded public CPU workflows. Validation is not
an independent audit. No original held-out audit row is copied into development.
