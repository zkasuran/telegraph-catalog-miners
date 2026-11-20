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
