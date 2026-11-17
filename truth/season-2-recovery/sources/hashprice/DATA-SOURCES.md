<!-- SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0 -->
# Data rights and source boundaries

Primary sources were read before implementation. Bibliography keys resolve through
`bibliography.json`. Research snapshots stay private. This publication includes
only our implementation, PublicNode consensus-fact fixtures and our measurements.

## Runtime source [PUBLICNODE]

Allnodes PublicNode is the sole runtime upstream. Its public Bitcoin endpoint is
`https://bitcoin-rpc.publicnode.com`. No key, account creation or paid archive tier
is used. All eight RPC requests are read-only queries.

> Subject to these Terms, you are hereby granted a non-exclusive, limited, non-transferable, freely revocable license to access and use the Service.

> You may access the Endpoints by automated means, including scripts, bots, indexers and AI agents, for development and production purposes. No prior authorization from us is required for such use.

> Blockchain data returned by the Endpoints is not Service Content. You may use it in your own applications and services, including displaying it to your users, but you may not resell access to the Endpoints or such data as a standalone product, such as an RPC or data API service, without our prior written consent.

The worker is a free Telegraph evaluation reference. Do not charge for endpoint
access or resell it as a standalone data API. The license is revocable. A later
terms change or withdrawal requires review. It is not a perpetual open-data grant.
HTTP 429 is respected. There is no bypass, proxy rotation or alternate paid tier.

## Second-service check [BC-DOC, BC-TERMS]

BlockCypher's Bitcoin mainnet block-by-height API was used for internal
corroboration only. Public docs invite unauthenticated read-only GET requests:

> We want everyone to try BlockCypher with as little friction as possible, which is why you don't need a token for any read-only GET calls.

The published Services Agreement, section 1.1, grants service access subject to
its conditions:

> BlockCypher hereby agrees that Customer has the non-exclusive right to access and use the services specified in the order form

The public GET invitation supports these internal reads. An order-form scope and
an express data-redistribution grant for this use are unsettled. The free tier
documents 3 requests/second and 100 requests/hour. Seven sequential GETs were
used. No BlockCypher response body enters the publication or runtime. PublicNode
is the licensed alternate selected for shipped data. Internal checker bodies stay
in local evidence. A numerical cross-check is not a provider license grant.

## Excluded services

Emzy and official Mempool [M1] have no express data-reuse grant established here.
Their disclaimer that the user must verify data is not permission to republish.
The old descriptor is not used. Blockstream [B1, B2] also has no express dataset
grant established here. Its prior verifier was not treated as data permission.

dRPC [DRPC] grants service access in section 1.1, but its live Bitcoin endpoint
returned `chain is not available on free plan, please upgrade to paid plan`.
It is excluded. Blockchain.com API restrictions and SoChain's prohibition on
competitive benchmarking were reviewed. Neither is a source for this artifact.

## Protocol sources

[CORE-POW], [CORE-TARGET], [CORE-REWARD], [CORE-TX] and [CORE-MERKLE] are Bitcoin
Core v29.0 primary source reads. No Bitcoin Core source or binary is embedded in
the runtime. The serialization and proof rules were implemented in our source.
The research-source license [CORE-LICENSE] grants:

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files

[D1] is the Bitcoin Wiki difficulty page, revision 70582. It states:

> Content is available under Creative Commons Attribution 3.0 unless otherwise noted.

Credit: Bitcoin Wiki contributors, Difficulty. The conventional `D * 2^32`
approximation is used because the recovered verified hashprice worker and truth
formula use it. It is not the exact uniform-hash probability `(target+1)/2^256`.
No copied Wiki code is included. The revision URL returned a 403 to the local
collector. The page was read successfully through the browser-fetch tool and its
existing local snapshot supplies private evidence. This limitation is recorded.

## Toolchain

Wrangler 4.163.0 is MIT OR Apache-2.0. ESLint 10.15.0 is MIT. Their granting
clause is `Permission is hereby granted, free of charge` with the rights to use,
copy, modify and distribute under the notice condition. Tool packages retain
their original notices and are installed from the pinned lockfile. Workerd is
Apache-2.0, whose section 2 grants a perpetual royalty-free copyright license to
reproduce and prepare derivative works. These are verification tools, not code
embedded in our worker. `npm audit` covers the complete toolchain lockfile.

The optional CI workflow pins GitHub's checkout, setup-node and setup-python
actions to full commit hashes [CI-CHECKOUT, CI-NODE, CI-PYTHON]. Their MIT licenses
grant `Permission is hereby granted, free of charge` under the retained notice
condition. No action implementation is copied into this worker.

## Own material

Response rules [CORE-RPC-INTERFACE] are read from Bitcoin Core v29.0 primary
documentation. JSON-RPC.ORG and the JSON-RPC Working Group specifications
[JSONRPC-1, JSONRPC-2] were read before compatibility changes. Their explicit
grant states that the documents may be used to implement JSON-RPC. No specification
body or third-party parser implementation is included in the worker. Attribution
is retained in NOTICE and the bibliography. This is independent implementation
of protocol facts, not a source fork.

`LICENSE` is the canonical LicenseRef-zkasuran-SAND-1.0 text, copied byte-for-byte.
`EVALUATION-PERMISSION.txt` gives Telegraph explicit storage, distribution and
operational permissions for our descriptor and free evaluation reference.
The worker's source and YAML carry SPDX notices. JSONL embeds the licensed YAML.
Bitcoin consensus facts are not asserted as our exclusive copyrighted property.
The original owner is zkasuran. No source fork or binary wrap is included.
