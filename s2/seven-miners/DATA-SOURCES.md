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

## Media input and output [MM-MEDIA]

The shipped PNG and two-second MP4 are owned rendered graphics created with
non-neural code. Full SAND terms are embedded in their metadata. No third-party
media or protected evaluation material enters these checks. FFmpeg is used as a
build tool; its executable or libraries are not redistributed in this package.
Other callers must supply media they are authorized to process with the provider.
The download is bounded and the exact fetched bytes are passed as a data URL.
The model's visual observations and confidence are uncalibrated, not independently
verified forensic findings. Audio is not processed.

[MM-MEDIA]: https://platform.minimax.io/docs/api-reference/text-chat-openai.md
[FFMPEG]: https://ffmpeg.org/legal.html

## X source discovery [X-POLICY] [MM-SEARCH]

Only post IDs and source links leave this service. The policy states:

<!-- voice-gate:off -->
> If you provide X Content to third parties, including downloadable datasets or via an API, you may only distribute Post IDs, Direct Message IDs, and/or User IDs (except as described below).
<!-- voice-gate:on -->

Default discovery uses actual MiniMax server-search result URLs, not model-invented
links. Each is confirmed through the public X oEmbed API. No embedded post text,
user object, protected data or full search snippet is redistributed. Direct paid
X API access remains disabled without account and spending authorization.
Deleted or unavailable posts are not returned after the publisher check.

[X-POLICY]: https://docs.x.com/developer-terms/agreement-and-policy
[X-OEMBED]: https://docs.x.com/x-for-websites/oembed-api
[MM-SEARCH]: https://platform.minimax.io/docs/guides/server-tools.md

## Parcel data [DHL]

DHL explicitly limits its tracking data to legitimate tracking purposes. It calls
that data confidential and requires prior written consent for other disclosure.
The demo key returns mocked tracking data. This integration does not treat that
demo as a live parcel. An actual account, actual public-status permission and a
current per-tracking-number grant are required before any request. No recipient
names, addresses, locations or proof-of-delivery are exposed. No real tracking
data was fetched or republished during the build. The tests use owned artificial
records only. This source remains unconfigured and live-unverified.

[DHL]: https://developer.dhl.com/api-reference/shipment-tracking

## Port observations [PORT-FEED] [HMAC]

This is a first-party signed-feed interface, not an assertion that a live port
data source has been acquired. A source's public-data licence must be read and
approved before configuration. The publisher must report mean completed anchorage
waiting hours, matching port identity and a recent observation. HMAC-SHA256
authenticates the exact payload bytes. It does not independently prove the physical
measurement. ETA differences, berth occupancy and unsupported units are rejected.
No real port observation was introduced during the build. Tests are artificial.

[PORT-FEED]: REQUIRED-SOURCES.json
[HMAC]: https://www.rfc-editor.org/rfc/rfc2104.html
