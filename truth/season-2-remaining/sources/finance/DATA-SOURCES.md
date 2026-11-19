# Finance source permissions

Only our source files and derived on-chain observations enter the ready bundles.
Raw issuer content, upstream software and protected evaluation material stay outside them.

## Fixed Nodies public Ethereum endpoint (77 and 136 current runtime)

Endpoint: https://ethereum-public.nodies.app
Primary terms: https://nodies.app/tos.txt
Primary access documentation: https://docs.nodies.app/rpc-services/public-endpoints.md
The primary clauses were read and recorded in `nodies/PERMISSIONS.json` before the first new RPC request. Original primary captures are `evidence/nodies/terms.body` and `evidence/nodies/public-endpoints.body`.

> Public endpoints are currently shared amongst all developers to use.

> You agree to use our Service only for lawful purposes.

> You agree to use our Service responsibly and within the bounds of the law, as well as these Terms.

> We own all intellectual property rights in the Service. You may not use our intellectual property rights without our prior written permission.

Scope: responsible lawful access to the documented shared public RPC service. We compute reserve and constant-current-rate APY values from public chain facts with our own code. No Nodies software, Service intellectual property or proprietary dataset is copied or redistributed. This is not a claimed software-copy licence or blanket dataset redistribution grant. The endpoint is fixed, with no alias probing or provider rotation. Shared service access can be limited or withdrawn.

## Historical PublicNode Endpoint data (77 and 136 preserved evidence)

Primary terms: https://www.publicnode.com/terms
Captured in `evidence/primary/publicnode-terms.body`.

> You may access the Endpoints by automated means, including scripts, bots, indexers and AI agents, for development and production purposes. No prior authorization from us is required for such use.

> Blockchain data returned by the Endpoints is not Service Content. You may use it in your own applications and services, including displaying it to your users, but you may not resell access to the Endpoints or such data as a standalone product, such as an RPC or data API service, without our prior written consent.

Scope: earlier observations and reference fixtures used PublicNode. Their original dates, URLs, hashes and attribution are retained. The new runtime uses Nodies only. These PublicNode terms remain the permission record for historical inputs, not a claim that PublicNode served the new requests. No independent financial assessment is inferred from a second RPC operator.

## Bitview confirmed Bitcoin balances (77 corroboration)

Primary terms: https://bitview.space/terms/
Captured in `evidence/primary/bitview-terms.body`.

> The data the services return, series descriptions included, and the charts they draw are under Creative Commons Attribution 4.0 (CC BY 4.0), as far as Bitview holds rights in them.

> You can copy, publish, change and build on the data and charts, for any purpose, commercial included, without asking.

Credit: Bitview (bitview.space), https://bitview.space/, CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.
Changes: subtract confirmed spent satoshis from funded satoshis, sum the disclosed address set and compare it with the on-chain attestation. Only chain balances and a block identity are used. Exchange prices, logos and arbitrary transaction content are excluded. This permission does not apply to WBTC's separate API.

## WBTC issuer discovery and API (77 research only)

Primary API documentation: https://docs.wbtc.network/resources/api-reference
Primary Website terms: https://www.wbtc.network/terms-of-service

The API documentation explicitly names verification of public reserve addresses as a typical use. Website terms require prior written permission for redistribution of Website materials. No blanket API-data redistribution grant was found.

The current reference uses the genuine on-chain WBTC PoR oracle attestation, read through the fixed Nodies public endpoint. Earlier PublicNode observations remain historical evidence. The handler does not proxy the issuer API or publish its financial body. The private 109-address corroboration states that custody designation comes from the preserved issuer disclosure. It does not claim to independently prove that designation or a complete liability ledger.

## Aave address metadata and APY method (136)

Commit-pinned references are listed in `primary-aave-registry.json` and `primary-aave-math.json`.
Primary grants: `evidence/primary/aave-registry-1.body` and `evidence/primary/aave-math-2.body`.

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

MIT, Copyright 2022 BGD Labs and Copyright (c) 2021 Aave, respectively. The package includes their licence notices. No Aave source module or model weight is copied. The current address facts and the per-second annualization method were checked against these primary sources. The on-chain Pool and DataProvider independently locate the same annual liquidityRate.

## Compound III rate model and convention (136)

Primary docs: https://docs.compound.finance/interest-rates/
Commit-pinned contract address and interface: `primary-compound.json`.
The read licence is in `evidence/primary/compound-3.body`:

> The Licensor hereby grants you the right to copy, modify, create derivative works, redistribute, and make non-production use of the Licensed Work.

The quoted BSL Change Date is the earlier of 2025-12-31 or its specified ENS date. It has passed for the inspected version, whose Change Licence is GPL v2 or later. Upstream source bytes are not distributed in our source bundle. Our own arithmetic implements the documented mathematical rate convention from separately observed public getter values.

Compound's docs and its public frontend report APR. Our `supply_apy` is explicitly the constant-current-rate, base-supply APY `(1 + per_second_rate / 1e18)^31536000 - 1`, multiplied by 10000 for bps. No reward-inclusive APY, realised return or provider-reported APY is claimed.

## dRPC transport check (136 private corroboration only)

Primary terms: https://drpc.org/terms-of-use
Captured in `evidence/primary/dRPC-terms.body`.

> The Client is granted a non-transferable, non-exclusive, non-sublicensable limited term worldwide right and license to access and use the Services.

This limited service permission supports the bounded private read-only contract comparison. Website data collection and competing-purpose restrictions are recorded separately. No dRPC Website data, software or raw API product is republished. The ready handler uses Nodies only; dRPC is a separate private proof path, never a runtime fallback. Multiple RPC operators corroborate transport/state, not independent yield assessments.

## CAISO general public materials and API restriction (92)

Primary terms: https://www.caiso.com/privacy-terms-of-use
Captured in `evidence/primary/caiso-terms.body`.

> Most of the materials and information contained on this Website and the California ISO mobile application were generated, compiled or assembled from materials and information that are freely available for public use consistent with the general policies of the Public Records Act (California Government Code section 6250, et seq.), as provided by California Public Utilities Code section 345.5, and may be used by you provided that you keep intact all copyright, trademark and other proprietary notices and that you credit the California ISO when using such materials and/or information.

Separate API restriction:

> You may not use or access the CAISO API for purposes of monitoring the availability, performance or functionality of any of CAISO's systems or services or for any other benchmarking or competitive purposes.

No CAISO API call was made by this role. A materially different independently hosted ERW archive contains the required historical DAM hour at three exact hubs. Its data are rounded to cents. The general CAISO grant is not represented as a blanket archive-publisher grant. A permitting archive-specific redistribution clause, corroboration of the original precision and a canonical node binding are still open. Candidate rows and their normalizer are private and unqueued.

## Binlist (116 research only)

Primary page: https://binlist.net/
The source describes a composite metadata service that can form data by assumptions about adjoining cards. `number.luhn` describes a card-number checksum scheme. It is not a checksum verdict for the supplied prefix. HTTP 400 for 00000000 is not evidence for `valid: false`.

No hosted-data redistribution grant or exact current eight-digit issuance guarantee was established. No Binlist response is distributed as a ready truth source. Our checksum counterexample uses mathematical test strings and is labelled accordingly. No real card, bank account or active standing is invented.

## Preserved local Worker verification tools

The local gate reuses Miniflare 5.20261116.0-beta and workerd 1.20261116.1 from the existing toolchain. No dependency was installed and no toolchain code is copied into the source bundles.

Primary Miniflare/Workers SDK grant: https://raw.githubusercontent.com/cloudflare/workers-sdk/main/LICENSE-MIT
Copyright (c) 2020 Cloudflare, Inc.

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The actual module is tested under the preserved Worker compatibility date 2025-06-01. Cloudflare's primary Request reference states that only follow/manual redirect modes are supported. The handler uses manual mode and refuses every redirect. Reference: https://developers.cloudflare.com/workers/runtime-apis/request/.

## Canonical block identity and RPC reply profile

Primary [FIN-EIP1898]: Charles Cooper, https://eips.ethereum.org/EIPS/eip-1898.

> Copyright and related rights waived via CC0.

This primary specification defines blockHash with requireCanonical for eth_call and explains why a before/after block-number check misses an ABA reorganization. The source uses the same canonical hash for all state queries. The new Nodies probe accepts the shape and refuses an unknown hash. Historical PublicNode and dRPC support probes are retained under their original identities. No numeric or latest fallback is used.

Primary [FIN-JSONRPC2]: JSON-RPC Working Group, https://www.jsonrpc.org/specification.

> This document and translations of it may be used to implement JSON-RPC, it may be copied and furnished to others, and derivative works that comment on or otherwise explain it or assist in its implementation may be prepared, copied, published and distributed, in whole or in part, without restriction of any kind, provided that the above copyright notice and this paragraph are included on all such copies and derivative works.

Copyright (C) 2007-2010 by the JSON-RPC Working Group. No source code is copied. The modern envelope requires the request's ID and exactly one result or error. This application uses literal numeric ID 1 and rejects duplicates, unknown top-level members and noncanonical ID spellings rather than repairing them.

Primary [FIN-HTTP-RETRY]: RFC 9110 section 10.2.3, https://www.rfc-editor.org/rfc/rfc9110.html#section-10.2.3. Retry-After is a delay in seconds or an HTTP-date. No code component is extracted. The read policy accepts nonnegative delta-seconds or canonical IMF-fixdate, then refuses advice that cannot fit the existing read/request budget.
