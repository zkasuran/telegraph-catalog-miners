# Sources and retained grants

## Original corpus

Parser diagnosis reuses only the exact owned frozen artifact from training run 44577682115 at source commit 0077ea0fab7c15cc0a026016fbf0e07591aec71c. Candidate, untrained and checkpoint bytes keep their existing terms and provenance. This is not new training or test data. Validation was measured at 50%; no independent audit score was exposed. Real parser stdout/stderr will be retained rather than replacing a failure with an expected result.

All Solidity source and reports in this experiment are newly authored AI-assisted work for zkasuran. There is no imported contract dataset and no human-review claim. Our original additions use LicenseRef-zkasuran-SAND-1.0. The full owned inventory is 152 sources across 38 grouped implementations. Vulnerable/fixed siblings and all identifier variants stay together. Training and validation bytes are unchanged. The 32 fresh audit groups cover distinct storage/authorization and arithmetic execution paths, not 32 people or additional renamed entities. Previous exposed audit designs 6 and 7 are not used in any split.

SmartBugs Curated was reviewed at `230e649123477eff332742a59a1c7cc6dc286cab`. Its README states: "All the contracts were obtained from public websites or using Etherscan and they retain their original licenses." No imported contract is cleared or included. Metadata Apache terms are not applied to Solidity files.

## Telegraph reference and wire

The pinned trainer's `_pair` supplies a default margin of 0.0. Its actual `PairCollator` indexes `p["margin"]`. Source inventories are not loader outputs, so the owned adapter supplies this field at the training boundary. It preserves source bytes, literal prompt/answer strings and provenance. The failure was observed before the first batch in GitHub run 44549236051, not inferred from a model-quality result. No audit score was exposed or used to select this correction.

Source revision: `1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd` at https://github.com/telegraphprotocol/telegraph-wasm-baseline.

Copyright (c) 2026 telegraphprotocol. MIT grant: "Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files" with rights to "use, copy, modify, merge, publish, distribute, sublicense" subject to retaining the copyright and permission notice. Full notice is in `licences.json` and all scorer/checkpoint exports.

The full pinned repository tree contains 100 blobs and is not truncated. `float_runtime/train_adapter.py` defines the literal render. `pkg/wasm/remote/remote.go`, `cmd/adapterexec/main.go` and `pkg/wasm/runtime/lora_runtime.go` define the actual batch contract and unprefixed adapter identity. A guessed separate modern tokenizer render file was absent from this kit. No broader claim about unpublished node source follows from that absence.

## Backbone

Answer.AI and LightOn ModernBERT-base revision `8949b909ec900327062f0ebf497f51aef5e6f0c8` at https://huggingface.co/answerdotai/ModernBERT-base.

The card states: "We release the ModernBERT model architectures, model weights, training codebase under the Apache 2.0 license." Section 2 grants a "perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license" to reproduce and prepare derivative works. Full Apache terms and attribution are retained in `licences.json` and all exported package/checkpoint bytes. No local model operation is authorized.

## Compiler and interpreter

Solidity 0.8.30 docs and source: https://github.com/ethereum/solidity/tree/v0.8.30. The GPL-3.0 compiler is an unchanged build tool, not incorporated into our source or scorer binary. Section 2 states: "You may make, run and propagate covered works that you do not convey, without conditions so long as your license otherwise remains in force." It states that output is covered only if the output constitutes a covered work. The corpus is original source, not copied compiler code. No compiler redistribution is proposed.

EthereumJS VM, util and common v10.1.0: https://github.com/ethereumjs/ethereumjs-monorepo/tree/%40ethereumjs%2Fvm%4010.1.0. These versions use MPL-2.0. Section 2.1 grants rights to "use, reproduce, make available, modify, display, perform, distribute". Section 3.3 permits a Larger Work under terms of our choice while retaining the MPL rights for Covered Software. These libraries are unchanged package-manager dependencies, not relicensed SAND source. The complete MPL text is retained as `MPL-2.0.txt`.

ethereum-cryptography 3.2.0 is an unchanged MIT dependency used only to compute ABI selectors. Copyright (c) 2021 Patricio Palladino, Paul Miller and ethereum-cryptography contributors. Its grant includes "use, copy, modify, merge, publish, distribute, sublicense" subject to retaining its copyright and permission notice. The original text is retained in `ethereum-cryptography-MIT.txt`. Transitive package integrity is locked in `package-lock.json`; `tmp` is pinned to 0.2.7 to remove known advisories. Dependencies are fetched from their registries, not copied into the proposed patch.

## Public live scope

Hosted workflow admission uses GitHub's documented `GITHUB_EVENT_PATH`, the full event webhook payload, plus the default repository context. GitHub documents that `GITHUB_*` and `RUNNER_*` defaults cannot be overwritten by workflow configuration. The exact repository identity and boolean `private: false` are required. This replaces the unauthenticated visibility lookup that returned a real 403 rate-limit error in run 44574642219. It changes the verification channel, not the public-only, model-quality or source-data gates. Reference: https://docs.github.com/en/actions/reference/workflows-and-actions/variables.

Keyless metadata only: https://devnode.telegraphprotocol.com/engine/v1/intents/SMART_CONTRACT_AUDIT_GENERAL. The source review read all 19 listed miners, not evaluation rows. Judgment and generic routing are confirmed. No miner provider credentials or calls were used. The owned CodeWire description explicitly accepts contract source or the whole question. The absence of typed schemas remains a coverage limit rather than a blanket scorer block.
