<!-- SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0 -->
# Source rights and scope

Primary clauses were read before using each service. Local snapshots and their
hashes are recorded in `research/sources.json`. Source keys are in `PROVENANCE.json`.
No third-party runtime or model dependency is included. No RPC credential is used.

## [PUBLICNODE] Allnodes PublicNode

Terms: https://www.publicnode.com/terms

> Subject to these Terms, you are hereby granted a non-exclusive, limited, non-transferable, freely revocable license to access and use the Service.

> You may access the Endpoints by automated means, including scripts, bots, indexers and AI agents, for development and production purposes. No prior authorization from us is required for such use.

> Blockchain data returned by the Endpoints is not Service Content. You may use it in your own applications and services, including displaying it to your users, but you may not resell access to the Endpoints or such data as a standalone product, such as an RPC or data API service, without our prior written consent.

This source is a free evaluation reference. It is not a paid or resold RPC/data API.
Deployment must retain that purpose. Access limits are respected. No IP rotation,
key rotation or bypass is included. Archive access requiring a token is refused.
PublicNode is used on Polygon and Arbitrum, at the exact URLs in `worker.mjs`.

## [DRPC] dRPC

Terms: https://drpc.org/terms-of-use (January 14, 2025).

> The Client is granted a non-transferable, non-exclusive, non-sublicensable limited term worldwide right and license to access and use the Services.

Excessive or abusive requests may cause suspension under section 2.5.

The first clause grants service access, not a licence to redistribute dRPC software
or website prose. This package serves factual public-chain quantities and our own
computed conversion. It contains no dRPC website content as answer material.
The public endpoints are expressly published in dRPC's own chain-list instructions.
The general account-registration language is recorded, not treated as a key grant.
Only the existing open Ethereum and Polygon endpoints are used.

## [NODIES] Nodies

Terms: https://nodies.app/tos.txt (May 19, 2023).

> You agree to use our Service only for lawful purposes. You must not use our Service in any way that is unlawful, or harms us or any other person or entity.

Public endpoint instructions:
https://docs.nodies.app/rpc-services/public-endpoints.md

> Public endpoints are currently shared amongst all developers to use.

These clauses authorize shared public service use subject to the terms. They do
not grant rights to Nodies software or website prose. Only public chain facts and
our conversion are returned. Nodies is used on Ethereum and Arbitrum. Backend
operator independence from another named service is not attested.

## [EIP1898] Protocol semantics

https://eips.ethereum.org/EIPS/eip-1898

> Copyright and related rights waived via CC0.

The worker implements this public specification without copying an implementation.
State calls use `blockHash` with `requireCanonical: true`. It never changes a failed
historical query to `latest` or another block. The two services' execution headers
are witnesses, not consensus proofs.

## Rejected implementation sources

Tenderly's current terms limit service use to the user's benefit and restrict
published evaluations. It is not a shipped upstream here. BlastAPI's terms URL
now serves a deprecation notice. No Blast code or service is inherited. Neither
source's successful response would settle its licensing.

## Outbound terms and evaluation permissions

Own implementation: `LicenseRef-zkasuran-SAND-1.0`, full text in `LICENSE`.
The current canonical licence was copied byte-for-byte, including its wording.
`EVALUATION-PERMISSION.txt` grants Telegraph the storage and operational rights
needed for the exact truth descriptor without granting competing miner reuse.
Public-chain facts are not claimed as our copyrighted software. Research snapshots
are private evidence, not the public source package. No protected pairs are copied.
