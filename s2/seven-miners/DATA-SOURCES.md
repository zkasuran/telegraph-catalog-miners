# Digital allocation inventory

SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0

The source is the owner's primary controlled digital allocation database, read at
request time. No external retailer, physical shipment or third-party inventory is
used. Stock quantity is on-hand allocation minus reserved allocation. A missing
SKU is an error, not zero stock. Read time and mutation time are separate fields.

The source modules are owned by zkasuran and already run at the stated endpoint.
Their grant is in LICENSE: "Calling the licensor's own deployed endpoints is not
restricted by this licence at all. They are a public service and this licence is
about the source."

The independent tests generate fresh inventory states. They do not load submitted
evaluation rows, protected material or benchmark answers.

The canonical SKU_IN_STOCK_GENERAL Intent currently lacks an owner-defined
request/response format. Registration is separate from canonical comparison,
ranking and on-chain answer delivery. This descriptor does not claim those gates
are ready.

## Generated chat answers

The chat route is an integrated application using the existing MiniMax subscription.
The Open Platform agreement states:

<!-- voice-gate:off -->
> As between you and us, and to the extent permitted by applicable laws, you retain your ownership rights in Client input and generated content.
<!-- voice-gate:on -->

It also prohibits independently reselling services outside integrated applications.
This route exposes a bounded Telegraph answer contract, not an API key or unrestricted
provider proxy. Its persistent budget allows twenty attempts per UTC day. Outputs
are marked AI-generated and are not independently verified facts or calibrated
forecasts. No web retrieval or submitted evaluation content is used.

Primary terms and API contract:
https://platform.minimax.io/docs/guides/terms-of-service
https://platform.minimax.io/docs/api-reference/text-chat-openai.md

A bounded real call returned MiniMax-M3 on 20 November 2026. The legacy quota API's
zero total count is not interpreted as a guarantee of unlimited usage. M Plan is
recommended for individual interactive use rather than production workloads.
