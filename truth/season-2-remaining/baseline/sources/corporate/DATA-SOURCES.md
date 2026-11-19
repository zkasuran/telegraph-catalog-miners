# Corporate source rights and semantics

SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0

## [GLEIF-TERMS]

Primary source: https://www.gleif.org/en/meta/lei-data-terms-of-use
Read 2026-11-17. Section II.3, verbatim:

> The data available through the Access Service are provided under the CC0 licence, see CC0 1.0 Universal (CC0 1.0).

Section II.1 enables individual records and entire sets or subsets. Section IV.1(b)
requires the offered Access Service and prohibits circumventing technical restrictions.
IV.2-3 prohibit implying GLEIF association or endorsement. No logo is used here.
The official API at https://api.gleif.org/api/v1/lei-records is the only upstream.
No private data, credentials or paid calls are used.

## [GLEIF-CDF]

Primary source:
https://www.gleif.org/en/lei-data/access-and-use-lei-data/level-1-data-lei-cdf-3-1-format
Read 2026-11-17. EntityStatus, verbatim:

> ACTIVE - As of the last report or update, the Legal Entity reported that it was legally registered and operating.

> INACTIVE - It has been determined that the Entity that was assigned the LEI is no longer legally registered and/or operating, whether as a result of business closure, acquisition by or merger with another (or new) Entity, or determination of illegitimacy.

RegistrationStatus LAPSED, verbatim:

> An LEI registration that has not been renewed by the NextRenewalDate and is not known by public sources to have ceased operation

Only entity.status maps to active. LEI renewal, conformity flags and good standing
are not equivalent. Freshness checks bound Golden Copy publication age to 48 hours,
not the age of the entity's last report. The response preserves both timestamps.
The source cannot certify a legal status changed after the last report. It never
maps EntityCreationDate to incorporated and never calls a tax identifier a company
number. registration_id is explicitly the LEI. Authority and registeredAs are evidence.

## [CC0]

Primary source: https://creativecommons.org/publicdomain/zero/1.0/legalcode.en
GLEIF reference data remains CC0-1.0. Our terms do not restrict those facts.

## Own code and descriptor

Original code and prose: LicenseRef-zkasuran-SAND-1.0, see LICENSE.
The descriptor grants Telegraph permission to store, copy and use its unmodified
bytes as evaluation material. Per-row provenance carries data rights and own terms.
No third-party code is vendored. Public node source is inspected as reference only.

## Recovery lineage

The g30 descriptor and acquisition workflow supplied the endpoint, filter names,
status semantics and HTTP sidecar pattern. Their flawed model-only entity selection
is replaced by explicit matching. Existing DataWire code uses attributes.entity.status
but selects the first name-search result without a jurisdiction check. That behavior
is not reused. No protected preference row enters the new bundle or probes.
