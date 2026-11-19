# DNS answer consistency scorer

SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0

This scorer checks whether a raw DNS response answers the requested name and record type. It also checks an explicit `lookup_ok` against DNS `Status`. Status 0 is true even with no answers or a null MX record. An observed nonzero status is false. Missing status, an unavailable lookup or a shaped boolean without evidence produces no score.

It does not authenticate the miner's DNS report. A fabricated but internally consistent response can pass. It cannot retrieve current DNS state. Email authentication posture, attachment safety and phishing are outside this task.

## Run without models

Run `bash verification/owners/email_security/verify.sh` from the standalone public checkout. The same source-only gate runs before model imports in the bounded public workflow. It bootstraps only hash-pinned verification tools into a temporary directory when needed. It never requires another owner namespace or a private checkout path.

The CLI reads an answer from standard input. Exit 0 means bound internal consistency. Exit 1 means mismatch or unavailable evidence. Exit 2 means invalid input. Its output states the evidence limit.

## Data and model boundary

The original fixture generator creates 224 authored zone groups with 448 reversed preference rows. Training has 128 groups, validation has 32 and audit has 64. Every zone and both answer/query orientations stay together. The audit uses a held-out JSON layout. All protocol fixtures and preferences are constructed by AI-assisted code. They are not observed DNS outcomes or human-reviewed annotations. They are not independent evaluation contributors.

Both requests in a group use the same two answer strings with opposite preferences. Any fixed answer-only scorer must average chance on the group. Immutable signed integer token-frequency coefficients fit on training only. They do not require floating-point refitting. Length, lexical overlap and the exact parser are reported separately. Matching query text is relevant evidence for this narrow task, not independent DNS truth.

An unmodified upstream dataset gate must pass before any remote model work. The remote candidate is fresh rank 8 on the pinned foundation. At most two epochs use training; validation loss selects the candidate. The audit is measured only after a candidate-state hash is sealed. The exact parser remains the authority for this bounded consistency rule. A neural model is not claimed to beat the parser. Each claimed format needs 80% overall accuracy, 70% per label, five points over its untrained initialization and ten points over the strongest length, token-length, overlap or answer-only shortcut. These gates are not lowered for publication or registration.

## Wire contract

The prompt is the literal upstream `InputsQuestion` string for the live request descriptions. The `wire_test.go` test invokes the original Go function, including mixed case, underscores and the double period resulting from a trailing root dot. Answers remain raw. The source defaults `AdapterInput` to raw unless a format opts into shaping.

The current public schema and pinned source support this build surface. The deployed judgment question builder, installed adapter-input mode and hidden qualification coverage remain unverified. Fresh node qualification is required before any canonical claim. Bare shaped booleans cannot support this scorer.

## Limits and release gate

No model was loaded or scored on the laptop. No package is marked qualified. The local data is a constructed diagnostic, not a source-held-out real-world benchmark. Reserved zones and deterministic seeds do not become genuine DNS observations. The two primary resolver controls are development evidence only and are excluded from every split.

Publication, bounded public CPU dispatch, endpoint activation and testnet registration use the installed automatic action adapter. No human review wait is part of this pipeline. The adapter keeps credentials and signing authority. Neither the owner nor the public workflow receives a signer or provider credential.

The workflow has separate train and serve modes. Training uses one standard ubuntu-24.04 CPU runner with a 150-minute ceiling. Serving uses one standard CPU runner with a 350-minute ceiling and an existing five-hour restricted-forwarding lease. Both have read-only repository permissions. No private key is uploaded. The public lease is constructed with the unchanged `common.lease_for` contract and binds the same combined report hash as candidate, runtime and source proof.

Failed checkpoints retain own SAND terms plus upstream Apache-2.0 and MIT grants. The complete audit uses the full package parser and actual pinned reference engine. It repeats eight score vectors in the same and a fresh executor, rejects three wrong identities and requires literal bare-hash echoes and the real node input-root expression. Only actual passing results can emit `registration-ready.json`. Native Torch scores are not substituted for reference-engine results.
