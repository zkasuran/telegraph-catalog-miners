# Weather source boundary

## [W1] NWS grant

Primary URL: https://www.weather.gov/disclaimer
Complete retrieved HTML and its digest are in sources/nws-disclaimer and sources/capture-manifest.json.

Exact grant: "The information on National Weather Service (NWS) Web pages are in the public domain, unless specifically noted otherwise, and may be used without charge for any lawful purpose so long as you do not: 1) claim it is your own (e.g., by claiming copyright for NWS information -- see below), 2) use it in a manner that implies an endorsement or affiliation with NOAA/NWS, or 3) modify its content and then present it as official government material."

Exact exclusion: "Third-party information and imagery are used under license by the individual third-party provider." The page directs the reader to the third party for further-use rights.

Our NWS observations are not subject to our copyright. No endorsement is claimed. Constructed wrong answers are separate records and are never presented as official data.

## [W2] GHCN-Daily source and semantics

Primary URL: https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt
Read the entire Version 3.35 document, not an inferred grant for its international compilation.

Exact source flag 1 definition: "CF6 (form F6) daily climate summaries from the U.S. National Weather Service".
Only this source is admitted. W, 0, X, D and every other source remain excluded. A public GHCN URL does not clear third-party inputs.

TMAX and TMIN are tenths of degrees Celsius. PRCP is tenths of millimetres. Missing is -9999. A blank quality flag means "did not fail any quality assurance check". That is not a guarantee of truth. Measurement flag T means trace. H, I and P cannot establish exact observed zero and are excluded.

Each selected station file is inspected in full. The full station metadata file is inspected to bind IDs, official station names and coordinates. The six stations were specified before retrieval. This is a six-station archive subset, not a representative sample of all weather or all GHCN stations.

Dates mean source station observation days. No UTC midnight conversion is invented. Observations ending after September 2026 are excluded from the useful table. These archived CF6-derived values can be revised and are not asserted to be final certified climate records. No forecast issue-time or past forecast exists in this source. The exact task requires recorded high temperature and does not require a forecast.

## [K1] Telegraph reference source

Revision: 1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd
Repository: https://github.com/telegraphprotocol/telegraph-wasm-baseline
Copyright (c) 2026 telegraphprotocol. MIT grant includes "use, copy, modify, merge, publish, distribute, sublicense" subject to retaining the full copyright and permission notice. The retrieved full notice is sources/kit-license.

Literal input rendering is verified against primary functions, not a normalized surrogate. Any extracted primary function stays under MIT. The source-card code additions use LicenseRef-zkasuran-SAND-1.0. The existing reviewed repository LICENSE is authoritative and must be retained with published additions.

## [M1] Remote model boundary

The existing shared training source pins Answer.AI and LightOn ModernBERT-base revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 under Apache-2.0. No weights were downloaded or executed locally. Before any controller model job, recheck the primary card and retain the full Apache terms plus the MIT kit notice. The new delta and reward head use the reviewed SAND terms. Failed checkpoints have the same terms boundary.

## Data authorship

Source observations: NWS CF6-derived archive facts retained with source flags, station, day and original file line.
Source annotations: QC and measurement flags supplied by the archive. They are not preference votes.
Preference answers: deterministic AI-assisted construction from those facts. Not human-reviewed. Not protected evaluation material and not uploaded to the evaluation pool.

## [L1] Lint tool

Ruff 0.14.5, copyright 2022 Charles Marsh, MIT. Read the installed wheel's full primary grant. It permits "use, copy, modify, merge, publish, distribute, sublicense" with the copyright and permission notice retained. We do not redistribute or modify its wheel. The wheel retains its complete upstream notices. The x86_64 manylinux wheel digest is pinned in lint-requirements.txt from the primary PyPI release metadata: https://pypi.org/pypi/ruff/0.14.5/json.
