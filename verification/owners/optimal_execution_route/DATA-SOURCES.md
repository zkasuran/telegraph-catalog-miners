<!-- SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0 -->
# Sources and grants

## [PN] PublicNode returned blockchain data

https://www.publicnode.com/terms

> You may access the Endpoints by automated means, including scripts, bots, indexers and AI agents, for development and production purposes. No prior authorization from us is required for such use.

> Blockchain data returned by the Endpoints is not Service Content. You may use it in your own applications and services, including displaying it to your users, but you may not resell access to the Endpoints or such data as a standalone product, such as an RPC or data API service, without our prior written consent.

The retained observations are used within this bounded quote verification application, not standalone RPC or data API resale. One provider supplied the original observations. No independent-provider or cryptographic state-proof claim is made. Full original web terms and RPC archives are not in this source publication.

## [SDK] Uniswap v2-sdk

Pin 5174518d5b50d21e6e920fe57e2db914550eddf9. Full original MIT licence and copyright notice are in upstream/UNISWAP-MIT.txt.

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

https://github.com/Uniswap/v2-sdk/blob/5174518d5b50d21e6e920fe57e2db914550eddf9/src/entities/pair.ts

The full source inventory was inspected: 30 entries with 26 blobs and truncated=false. The fixed quote arithmetic follows fee-inclusive integer floors. The single 100-to-98 upstream fixture is a labelled arithmetic regression, not an observed swap. Our two-path search does not include split allocations or all venues.

## [KIT] Telegraph primary caller and executor

Current main pin 1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd was independently re-read. The complete tree has 128 entries. Relevant source inventory and source fingerprints are in source-manifest.json. Full original MIT licence: upstream/TELEGRAPH-MIT.txt, Copyright (c) 2026 telegraphprotocol. The same quoted MIT permission and notice-retention clauses above apply.

https://github.com/telegraphprotocol/telegraph-wasm-baseline/blob/1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd/pkg/scoring/responseparser/onchain.go
https://github.com/telegraphprotocol/telegraph-wasm-baseline/blob/1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd/cmd/adapterexec/main.go

Unchanged primary question and handler functions are compiled in isolated source-only plumbing. The handler uses a labelled mock scorer, never a model. Its input_root hashes received token lists, not independent chain evidence. Bare numeric truth still needs separately trusted state.

## [LIVE] Current public definition

https://devnode.telegraphprotocol.com/engine/v1/intents/OPTIMAL_EXECUTION_ROUTE_GENERAL

Only the task projection is retained. Miner records, protected rows and source web pages are absent. The actual live class is judgment. The fixed request declares pair=ETH/USDC and no block, state or objective. A missing miner schema is not the reason for the gate.

## [TOOLS] Passive source checkers

Pyflakes 3.4.0: https://pypi.org/pypi/pyflakes/3.4.0/json, MIT, no dependencies. Its pinned wheel is named in requirements-source.lock.
PyYAML 6.0.3: https://raw.githubusercontent.com/yaml/pyyaml/6.0.3/LICENSE, MIT, Copyright (c) 2017-2021 Ingy döt Net and Copyright (c) 2006-2016 Kirill Simonov. Its licence contains the quoted permission and retention clauses above. The hash-locked source gate supports Python 3.11 and 3.12 on public Linux x86_64. Both wheel hashes were independently verified against the primary release metadata. Installed tool bytes are not published.

## Data boundaries

Source code, observed state and constructed adversarial controls are distinct. Original exposed observations remain the same regression fixtures. Training, validation and independent audit allocations are all absent. No synthetic preference is called human-reviewed and no fixture count is an independence count. New additions use SAND with retained upstream grants.

## [UPLOAD] GitHub artifact transport

The pinned action's primary input contract is https://raw.githubusercontent.com/actions/upload-artifact/ea165f8d65b6e75b540449e92b4886f43607fa02/action.yml.

> If false, hidden files will be excluded from the artifact.

Its default is false. The source pipeline therefore stages only a closed list of report/log/licence assets into a normal directory. It does not enable broad hidden-file upload or copy tool/compiler caches. A source import guard independently rejects neural library imports in local and public source-only gates. That guard is our own source code and needs no model or provider credentials.
